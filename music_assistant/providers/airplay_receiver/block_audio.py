"""Reader for shairport-sync's blockaudio backend: blocks of audio with their play times."""

from __future__ import annotations

import asyncio
import os
import struct
import zlib
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from logging import Logger

    from music_assistant.helpers.live_timing import LiveTiming

# The record header, version 1 (shairport-sync's ADVANCED TOPICS/BlockAudio.md): magic,
# version, header length, payload length, sequence, header CRC-32, flags, reserved, format,
# RTP timestamp, play time (ns). Native byte order: the pipe never leaves this machine.
HEADER = struct.Struct("=4sHHIIIHHIIQ")
MAGIC = b"SSBA"
CRC_OFFSET = 16
FLAG_TIMED = 1
FLAG_FLUSH = 2
FLAG_STOP = 4
# a header longer than this is garbage, not a later version
MAX_HEADER_LENGTH = 1024
# shairport-sync's sps_format_t for little-endian PCM of each bit depth
SPS_FORMATS = {16: 3, 24: 7, 32: 9}
# the lead-in after a session start can be well over a second in a single record
MAX_RECORD_SECONDS = 10
# what the reader holds for a consumer that falls behind, before dropping the oldest audio
MAX_QUEUED_SECONDS = 10
READ_SIZE = 64 * 1024


class _Header(NamedTuple):
    magic: bytes
    version: int
    header_length: int
    payload_length: int
    sequence: int
    crc: int
    flags: int
    reserved: int
    format: int
    rtp: int
    play_ns: int


def encoded_format(sample_rate: int, channels: int, bit_depth: int) -> int:
    """
    Return shairport-sync's encoded output format for little-endian PCM.

    :param sample_rate: Frames per second.
    :param channels: Number of channels.
    :param bit_depth: Bits per sample: 16, 24 (packed in 3 bytes) or 32.
    """
    return (channels << 25) | ((sample_rate // 2) << 6) | SPS_FORMATS[bit_depth]


@dataclass(frozen=True, slots=True)
class PipeRecord:
    """One record from the pipe: audio, or a flush/stop marker without audio."""

    flags: int
    # ns on shairport-sync's clock at which the first frame should be played; 0 when untimed
    play_ns: int
    pcm: bytes
    # records were lost right before this one, so what was received before doesn't continue
    discontinuity: bool = False


class BlockAudioParser:
    """Split the pipe's byte stream into records."""

    def __init__(self, sample_rate: int, channels: int, bit_depth: int) -> None:
        """
        Initialize the parser.

        :param sample_rate: The sample rate shairport-sync is configured to output.
        :param channels: The number of channels it is configured to output.
        :param bit_depth: The bits per sample it is configured to output.
        """
        self.sample_rate = sample_rate
        self.bytes_per_frame = channels * ((bit_depth + 7) // 8)
        self.format = encoded_format(sample_rate, channels, bit_depth)
        # records lost to a resync or a gap in the sequence, after the first record
        self.discontinuities = 0
        # formats of records skipped because they aren't the configured one
        self.rejected_formats: set[int] = set()
        self._buffer = bytearray()
        self._sequence: int | None = None
        self._lost = False

    def feed(self, data: bytes) -> list[PipeRecord]:
        """
        Parse the next bytes read from the pipe and return the records they complete.

        :param data: Bytes read from the pipe.
        """
        self._buffer += data
        records: list[PipeRecord] = []
        while len(self._buffer) >= HEADER.size:
            header = _Header._make(HEADER.unpack_from(self._buffer))
            header_length = header.header_length
            if header.magic != MAGIC or not HEADER.size <= header_length <= MAX_HEADER_LENGTH:
                if not self._resync():
                    break
                continue
            if len(self._buffer) < header_length:
                break
            if not self._is_valid(header, header_length):
                if not self._resync():
                    break
                continue
            size = header_length + header.payload_length
            # look before the payload is complete: if this record was cut short, the bytes
            # that would complete it may be minutes away
            cut = self._find_cut(header_length, min(size, len(self._buffer)))
            if cut is not None and cut != -1:
                # shairport-sync cut this record short and wrote the next one after it
                del self._buffer[:cut]
                self._lose()
                continue
            if cut == -1 or len(self._buffer) < size:
                break
            if (record := self._take_record(header, header_length, size)) is not None:
                records.append(record)
        return records

    def _is_valid(self, header: _Header, header_length: int) -> bool:
        if header.version == 0 or not self._crc_matches(0, header_length, header.crc):
            return False
        payload_length = header.payload_length
        if header.flags & (FLAG_FLUSH | FLAG_STOP):
            return payload_length == 0
        return (
            0 < payload_length <= MAX_RECORD_SECONDS * self.sample_rate * self.bytes_per_frame
            and payload_length % self.bytes_per_frame == 0
            and bool(header.play_ns) == bool(header.flags & FLAG_TIMED)
        )

    def _crc_matches(self, offset: int, header_length: int, crc: int) -> bool:
        checked = bytearray(self._buffer[offset : offset + header_length])
        checked[CRC_OFFSET : CRC_OFFSET + 4] = bytes(4)
        return zlib.crc32(checked) == crc

    def _find_cut(self, start: int, end: int) -> int | None:
        """
        Return where a header starts inside the payload in ``start:end``, or None if none does.

        Returns -1 when that can't be told before more data arrives.
        """
        # the pipe stays open while shairport-sync closes and reopens its end, so a record
        # it cut short shows no end of file, only the next record inside its payload
        search_end = end + len(MAGIC) - 1
        index = self._buffer.find(MAGIC, start, search_end)
        while index != -1:
            if len(self._buffer) < index + HEADER.size:
                return -1
            header = _Header._make(HEADER.unpack_from(self._buffer, index))
            header_length = header.header_length
            if HEADER.size <= header_length <= MAX_HEADER_LENGTH:
                if len(self._buffer) < index + header_length:
                    return -1
                if self._crc_matches(index, header_length, header.crc):
                    return index
            index = self._buffer.find(MAGIC, index + 1, search_end)
        # a header starting in the last bytes, with the rest of its magic not read yet, goes
        # unnoticed: waiting for more would hold back every record ending in a prefix of it
        return None

    def _take_record(self, header: _Header, header_length: int, size: int) -> PipeRecord | None:
        """Remove a complete record from the buffer; None if it isn't for us."""
        pcm = bytes(self._buffer[header_length:size])
        del self._buffer[:size]
        if self._sequence is not None and header.sequence != (self._sequence + 1) & 0xFFFFFFFF:
            self._lose()
        self._sequence = header.sequence
        if pcm and header.format != self.format:
            self.rejected_formats.add(header.format)
            self._lose()
            return None
        discontinuity, self._lost = self._lost, False
        return PipeRecord(header.flags, header.play_ns, pcm, discontinuity)

    def _resync(self) -> bool:
        """Drop bytes up to the next candidate header; False when more data is needed."""
        self._lose()
        index = self._buffer.find(MAGIC, 1)
        if index == -1:
            # keep a tail that may hold the start of the magic
            del self._buffer[: len(self._buffer) - len(MAGIC) + 1]
            return False
        del self._buffer[:index]
        return True

    def _lose(self) -> None:
        # a reader starting in the middle of the stream has nothing to lose yet
        if self._sequence is not None and not self._lost:
            self._lost = True
            self.discontinuities += 1


class _Consumer:
    """The audio queued for the stream that currently reads the pipe."""

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max_frames
        self.records: deque[tuple[bytes, int | None, int]] = deque()
        self.frames = 0
        self.dropped_frames = 0
        self.closed = False
        self._wakeup = asyncio.Event()

    def put(self, pcm: bytes, play_ns: int | None, frames: int) -> None:
        self.records.append((pcm, play_ns, frames))
        self.frames += frames
        while self.frames > self.max_frames:
            *_, dropped_frames = self.records.popleft()
            self.frames -= dropped_frames
            self.dropped_frames += dropped_frames
        self._wakeup.set()

    def clear(self) -> None:
        self.records.clear()
        self.frames = 0

    def close(self) -> None:
        self.closed = True
        self._wakeup.set()

    async def get(self) -> tuple[bytes, int | None, int] | None:
        while not self.records:
            if self.closed:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()
        record = self.records.popleft()
        self.frames -= record[2]
        return record


class BlockAudioReader:
    """
    Keep shairport-sync's blockaudio pipe open and drained for the lifetime of the daemon.

    Audio before a session's first timed frame (the lead-in silence) is dropped; a
    stream that attaches gets the audio from then on, along with its play times.
    """

    def __init__(
        self, path: str, sample_rate: int, channels: int, bit_depth: int, logger: Logger
    ) -> None:
        """
        Initialize the reader.

        :param path: Path of the audio FIFO.
        :param sample_rate: The sample rate shairport-sync is configured to output.
        :param channels: The number of channels it is configured to output.
        :param bit_depth: The bits per sample it is configured to output.
        :param logger: Logger instance.
        """
        self.path = path
        self.logger = logger
        self._parser = BlockAudioParser(sample_rate, channels, bit_depth)
        self._fd: int | None = None
        self._consumer: _Consumer | None = None
        self._timing: LiveTiming | None = None
        # play time of the frame after the last timed audio; None until a session's first
        self._next_play_ns: int | None = None
        self._discontinuities_logged = 0
        self._formats_logged: set[int] = set()

    async def start(self) -> None:
        """Open the pipe and start draining it."""
        # O_RDWR keeps a writer attached so the read end never reaches EOF (see
        # MetadataReader) and shairport-sync always finds a reader when it opens the pipe
        self._fd = await asyncio.to_thread(os.open, self.path, os.O_RDWR | os.O_NONBLOCK)
        asyncio.get_running_loop().add_reader(self._fd, self._on_readable)

    async def stop(self) -> None:
        """Stop reading, close the pipe and end the attached stream."""
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
        if self._fd is not None:
            asyncio.get_running_loop().remove_reader(self._fd)
            with suppress(OSError):
                os.close(self._fd)
            self._fd = None

    async def iter_audio(self, timing: LiveTiming) -> AsyncGenerator[bytes]:
        """
        Yield the received audio until the reader stops or another stream attaches.

        :param timing: Receives the play time of every frame yielded.
        """
        if self._consumer is not None:
            self._consumer.close()
        consumer = self._consumer = _Consumer(MAX_QUEUED_SECONDS * self._parser.sample_rate)
        self._timing = timing
        try:
            while (record := await consumer.get()) is not None:
                pcm, play_ns, frames = record
                if play_ns is not None:
                    timing.add_audio(frames, play_ns)
                yield pcm
        finally:
            if consumer.dropped_frames:
                self.logger.warning(
                    "AirPlay audio stream fell behind, dropped %.1f s of audio",
                    consumer.dropped_frames / self._parser.sample_rate,
                )
            if self._consumer is consumer:
                self._consumer = None
                self._timing = None

    def _on_readable(self) -> None:
        assert self._fd is not None
        try:
            data = os.read(self._fd, READ_SIZE)
        except BlockingIOError:
            return
        except OSError as err:
            self.logger.debug("Error reading from the audio pipe: %s", err)
            return
        for record in self._parser.feed(data):
            self._handle_record(record)
        self._log_parser_problems()

    def _handle_record(self, record: PipeRecord) -> None:
        if record.discontinuity or record.flags & (FLAG_FLUSH | FLAG_STOP):
            self._flush()
            if not record.pcm:
                return
        frames = len(record.pcm) // self._parser.bytes_per_frame
        # untimed audio after timed audio fills a gap (missing packets) and plays on from
        # it; untimed audio before it is the lead-in silence nobody needs
        play_ns = record.play_ns if record.flags & FLAG_TIMED else self._next_play_ns
        if play_ns is None:
            return
        self._next_play_ns = play_ns + frames * 1_000_000_000 // self._parser.sample_rate
        if self._consumer is not None:
            self._consumer.put(record.pcm, play_ns, frames)

    def _flush(self) -> None:
        """Drop what is queued: the sender discarded it, or it no longer continues."""
        # the next audio starts a new timeline
        self._next_play_ns = None
        if self._consumer is not None:
            self._consumer.clear()
        if self._timing is not None:
            self._timing.new_epoch()

    def _log_parser_problems(self) -> None:
        parser = self._parser
        if parser.discontinuities > self._discontinuities_logged:
            self._discontinuities_logged = parser.discontinuities
            self.logger.warning(
                "Lost audio records from shairport-sync (%s times), restarting the timeline",
                parser.discontinuities,
            )
        for fmt in parser.rejected_formats - self._formats_logged:
            self._formats_logged.add(fmt)
            self.logger.error(
                "shairport-sync sends audio in format 0x%08x instead of the configured 0x%08x",
                fmt,
                parser.format,
            )
