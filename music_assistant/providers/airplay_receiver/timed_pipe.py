"""Reader for shairport-sync's audio pipe with timing headers (pipe.timing_headers)."""

from __future__ import annotations

import asyncio
import os
import struct
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from logging import Logger

    from music_assistant.helpers.live_timing import LiveTiming

# magic, version, flags, frames, rtp timestamp, should-be time (ns), rate, bytes per frame
HEADER = struct.Struct("<IHHIIQII")
HEADER_MAGIC = 0x48545053
HEADER_MAGIC_BYTES = HEADER_MAGIC.to_bytes(4, "little")
HEADER_VERSION = 1
FLAG_TIMED = 1
FLAG_FLUSH = 2
FLAG_STOP = 4
# an older shairport-sync writes plain PCM; give up looking for headers after this much
RAW_DETECT_BYTES = 64 * 1024
# the lead-in after a session start can be well over a second in a single record
MAX_RECORD_SECONDS = 10
# what the reader holds for a consumer that falls behind, before dropping the oldest audio
MAX_QUEUED_SECONDS = 10
READ_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class PipeRecord:
    """One record from the pipe: audio, or a flush/stop marker without audio."""

    flags: int
    # CLOCK_MONOTONIC_RAW ns at which the first frame should be played; 0 when untimed
    play_ns: int
    pcm: bytes


class TimedPipeParser:
    """Split the audio pipe's byte stream into records, falling back to plain PCM."""

    def __init__(self, sample_rate: int, bytes_per_frame: int) -> None:
        """
        Initialize the parser.

        :param sample_rate: The sample rate shairport-sync is configured to output.
        :param bytes_per_frame: Bytes per frame of that output.
        """
        self.sample_rate = sample_rate
        self.bytes_per_frame = bytes_per_frame
        # None until the first valid header (timed) or RAW_DETECT_BYTES without one (raw)
        self.timed: bool | None = None
        self.resyncs = 0
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[PipeRecord]:
        """
        Parse the next bytes read from the pipe and return the records they complete.

        :param data: Bytes read from the pipe.
        """
        if self.timed is False:
            return [PipeRecord(0, 0, data)]
        self._buffer += data
        records: list[PipeRecord] = []
        while len(self._buffer) >= HEADER.size:
            header = HEADER.unpack_from(self._buffer)
            if not self._is_valid(header):
                if not self._resync():
                    break
                continue
            self.timed = True
            _, _, flags, frames, _, play_ns, _, _ = header
            size = HEADER.size + frames * self.bytes_per_frame
            if len(self._buffer) < size:
                break
            records.append(PipeRecord(flags, play_ns, bytes(self._buffer[HEADER.size : size])))
            del self._buffer[:size]
        if self.timed is None and len(self._buffer) >= RAW_DETECT_BYTES:
            self.timed = False
            records.append(PipeRecord(0, 0, bytes(self._buffer)))
            self._buffer.clear()
        return records

    def _is_valid(self, header: tuple[int, ...]) -> bool:
        magic, version, flags, frames, _, play_ns, rate, bytes_per_frame = header
        if (
            magic != HEADER_MAGIC
            or version != HEADER_VERSION
            or flags & ~(FLAG_TIMED | FLAG_FLUSH | FLAG_STOP)
            or rate != self.sample_rate
            or bytes_per_frame != self.bytes_per_frame
        ):
            return False
        if flags & (FLAG_FLUSH | FLAG_STOP):
            return frames == 0
        return 0 < frames <= MAX_RECORD_SECONDS * rate and bool(play_ns) == bool(flags & FLAG_TIMED)

    def _resync(self) -> bool:
        """Drop bytes up to the next candidate header; False when more data is needed."""
        index = self._buffer.find(HEADER_MAGIC_BYTES, 1)
        if self.timed:
            self.resyncs += 1
        if index == -1:
            if self.timed:
                # keep a tail that may hold the start of the magic
                del self._buffer[: len(self._buffer) - len(HEADER_MAGIC_BYTES) + 1]
            # before the first header the bytes are kept, in case the stream turns out raw
            return False
        del self._buffer[:index]
        return True


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


class TimedPipeReader:
    """
    Keep shairport-sync's audio pipe open and drained for the lifetime of the daemon.

    Audio before a session's first timed frame (the lead-in silence) is dropped; a
    stream that attaches gets the audio from then on, along with its play times.
    """

    def __init__(self, path: str, sample_rate: int, bytes_per_frame: int, logger: Logger) -> None:
        """
        Initialize the reader.

        :param path: Path of the audio FIFO.
        :param sample_rate: The sample rate shairport-sync is configured to output.
        :param bytes_per_frame: Bytes per frame of that output.
        :param logger: Logger instance.
        """
        self.path = path
        self.logger = logger
        self._parser = TimedPipeParser(sample_rate, bytes_per_frame)
        self._fd: int | None = None
        self._consumer: _Consumer | None = None
        self._timing: LiveTiming | None = None
        # play time of the frame after the last timed audio; None until a session's first
        self._next_play_ns: int | None = None
        self._resyncs_logged = 0

    @property
    def timed(self) -> bool:
        """Return whether shairport-sync writes timing headers (False until known)."""
        return bool(self._parser.timed)

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
        if self._parser.resyncs > self._resyncs_logged:
            self._resyncs_logged = self._parser.resyncs
            self.logger.warning(
                "Lost track of the timing headers in the audio pipe (%s times)",
                self._parser.resyncs,
            )

    def _handle_record(self, record: PipeRecord) -> None:
        consumer = self._consumer
        if not self._parser.timed:
            if consumer is not None:
                consumer.put(record.pcm, None, len(record.pcm) // self._parser.bytes_per_frame)
            return
        if record.flags & (FLAG_FLUSH | FLAG_STOP):
            # what is still queued was flushed by the sender, and the next audio starts a
            # new timeline
            self._next_play_ns = None
            if consumer is not None:
                consumer.clear()
            if self._timing is not None:
                self._timing.new_epoch()
            return
        frames = len(record.pcm) // self._parser.bytes_per_frame
        # untimed audio after timed audio fills a gap (missing packets) and plays on from
        # it; untimed audio before it is the lead-in silence nobody needs
        play_ns = record.play_ns if record.flags & FLAG_TIMED else self._next_play_ns
        if play_ns is None:
            return
        self._next_play_ns = play_ns + frames * 1_000_000_000 // self._parser.sample_rate
        if consumer is not None:
            consumer.put(record.pcm, play_ns, frames)
