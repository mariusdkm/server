"""Tests for the reader of shairport-sync's blockaudio pipe."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import zlib
from typing import TYPE_CHECKING

import pytest

from music_assistant.helpers.live_timing import LiveTiming
from music_assistant.providers.airplay_receiver.block_audio import (
    CRC_OFFSET,
    FLAG_FLUSH,
    FLAG_STOP,
    FLAG_TIMED,
    HEADER,
    MAGIC,
    MAX_QUEUED_SECONDS,
    BlockAudioParser,
    BlockAudioReader,
    PipeRecord,
    encoded_format,
)

if TYPE_CHECKING:
    from pathlib import Path

RATE = 44100
CHANNELS = 2
BIT_DEPTH = 16
BYTES_PER_FRAME = 4
FORMAT = encoded_format(RATE, CHANNELS, BIT_DEPTH)
SECOND_NS = 1_000_000_000


class _Writer:
    """Writes records the way shairport-sync's blockaudio backend does."""

    def __init__(self) -> None:
        self.sequence = 0

    def record(
        self,
        flags: int,
        frames: int = 0,
        play_ns: int = 0,
        fill: int = 1,
        fmt: int = FORMAT,
        extra_header: bytes = b"",
    ) -> bytes:
        payload = bytes([fill]) * frames * BYTES_PER_FRAME
        header_length = HEADER.size + len(extra_header)
        header = bytearray(
            HEADER.pack(
                MAGIC,
                1,
                header_length,
                len(payload),
                self.sequence,
                0,
                flags,
                0,
                fmt,
                1234 if flags & FLAG_TIMED else 0,
                play_ns,
            )
            + extra_header
        )
        header[CRC_OFFSET : CRC_OFFSET + 4] = zlib.crc32(header).to_bytes(4, sys.byteorder)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return bytes(header) + payload


def _parser() -> BlockAudioParser:
    return BlockAudioParser(RATE, CHANNELS, BIT_DEPTH)


def _stream() -> bytes:
    writer = _Writer()
    return (
        writer.record(0, 10, fill=0)
        + writer.record(FLAG_TIMED, 352, SECOND_NS)
        + writer.record(FLAG_FLUSH)
        + writer.record(FLAG_TIMED, 352, 2 * SECOND_NS, fill=2)
        + writer.record(FLAG_FLUSH | FLAG_STOP)
    )


EXPECTED = [
    PipeRecord(0, 0, bytes(40)),
    PipeRecord(FLAG_TIMED, SECOND_NS, b"\x01" * 1408),
    PipeRecord(FLAG_FLUSH, 0, b""),
    PipeRecord(FLAG_TIMED, 2 * SECOND_NS, b"\x02" * 1408),
    PipeRecord(FLAG_FLUSH | FLAG_STOP, 0, b""),
]


def test_encoded_format_matches_shairport_sync() -> None:
    """S16_LE, 44.1 kHz, stereo as shairport-sync encodes it."""
    assert FORMAT == (2 << 25) | (22050 << 6) | 3


@pytest.mark.parametrize("split", range(0, len(_stream()), 7))
def test_records_split_across_reads(split: int) -> None:
    """Records come out whole wherever the reads split them."""
    stream = _stream()
    parser = _parser()
    records = parser.feed(stream[:split]) + parser.feed(stream[split:])
    assert records == EXPECTED
    assert parser.discontinuities == 0


def test_reading_from_the_middle_of_a_record() -> None:
    """A reader that starts mid-stream finds the next record and has lost nothing yet."""
    stream = _stream()
    parser = _parser()
    second = len(EXPECTED[0].pcm) + HEADER.size
    assert parser.feed(stream[second - 7 :]) == EXPECTED[1:]
    assert parser.discontinuities == 0


def test_resync_after_garbage() -> None:
    """Bytes that are no record are skipped, and the next record says records were lost."""
    writer = _Writer()
    parser = _parser()
    parser.feed(writer.record(FLAG_TIMED, 10, SECOND_NS))
    records = parser.feed(
        b"\x00" * 100 + MAGIC + b"\x00" * 50 + writer.record(FLAG_TIMED, 352, SECOND_NS)
    )
    assert records == [PipeRecord(FLAG_TIMED, SECOND_NS, b"\x01" * 1408, discontinuity=True)]
    assert parser.discontinuities == 1


def test_header_with_a_wrong_crc_is_skipped() -> None:
    """A header whose CRC doesn't match is garbage, however plausible it looks."""
    writer = _Writer()
    parser = _parser()
    parser.feed(writer.record(FLAG_TIMED, 10, SECOND_NS))
    broken = bytearray(writer.record(FLAG_TIMED, 352, 5 * SECOND_NS))
    broken[32] ^= 0xFF  # the play time
    records = parser.feed(bytes(broken) + writer.record(FLAG_FLUSH))
    assert records == [PipeRecord(FLAG_FLUSH, 0, b"", discontinuity=True)]


def test_gap_in_the_sequence_is_a_discontinuity() -> None:
    """A record that doesn't follow the one before says records were lost."""
    writer = _Writer()
    parser = _parser()
    parser.feed(writer.record(FLAG_TIMED, 10, SECOND_NS))
    writer.sequence += 3
    assert parser.feed(writer.record(FLAG_TIMED, 10, 2 * SECOND_NS)) == [
        PipeRecord(FLAG_TIMED, 2 * SECOND_NS, b"\x01" * 40, discontinuity=True)
    ]
    assert parser.discontinuities == 1


def test_sequence_wraps_around() -> None:
    """The sequence number wraps around without counting as a gap."""
    writer = _Writer()
    writer.sequence = 0xFFFFFFFF
    parser = _parser()
    parser.feed(writer.record(FLAG_TIMED, 10, SECOND_NS) + writer.record(FLAG_FLUSH))
    assert writer.sequence == 1
    assert parser.discontinuities == 0


def test_longer_header_of_a_later_version_is_read() -> None:
    """Fields a later version appends to the header are skipped."""
    writer = _Writer()
    parser = _parser()
    records = parser.feed(writer.record(FLAG_TIMED, 352, SECOND_NS, extra_header=b"\xab" * 8))
    assert records == [EXPECTED[1]]


def test_audio_in_another_format_is_skipped() -> None:
    """A record in another format than the configured one is dropped, as lost audio."""
    writer = _Writer()
    parser = _parser()
    other = encoded_format(48000, CHANNELS, BIT_DEPTH)
    records = parser.feed(
        writer.record(FLAG_TIMED, 352, SECOND_NS)
        + writer.record(FLAG_TIMED, 352, 2 * SECOND_NS, fmt=other)
        + writer.record(FLAG_TIMED, 352, 3 * SECOND_NS)
    )
    assert [record.play_ns for record in records] == [SECOND_NS, 3 * SECOND_NS]
    assert records[1].discontinuity
    assert parser.rejected_formats == {other}


@pytest.mark.parametrize("kept", [0, 10, 1400, 1405])
def test_record_cut_short_is_dropped(kept: int) -> None:
    """A record cut short ends where the next record starts, even inside its payload."""
    writer = _Writer()
    parser = _parser()
    parser.feed(writer.record(FLAG_TIMED, 10, SECOND_NS))
    cut = writer.record(FLAG_TIMED, 352, 2 * SECOND_NS)[: HEADER.size + kept]
    records = parser.feed(cut + writer.record(FLAG_TIMED, 352, 3 * SECOND_NS, fill=3))
    assert records == [PipeRecord(FLAG_TIMED, 3 * SECOND_NS, b"\x03" * 1408, discontinuity=True)]


def test_record_cut_short_does_not_hold_back_the_next() -> None:
    """The next record comes out right away, not once the cut record's length has arrived."""
    writer = _Writer()
    parser = _parser()
    parser.feed(writer.record(FLAG_TIMED, 10, SECOND_NS))
    cut = writer.record(FLAG_TIMED, 5 * RATE, 2 * SECOND_NS)[: HEADER.size + 1000]
    assert parser.feed(cut) == []
    assert parser.feed(writer.record(FLAG_FLUSH)) == [PipeRecord(FLAG_FLUSH, 0, b"", True)]


def test_audio_that_looks_like_a_header_is_audio() -> None:
    """The magic in the audio, without a header around it, doesn't cut the record."""
    writer = _Writer()
    parser = _parser()
    record = bytearray(writer.record(FLAG_TIMED, 352, SECOND_NS))
    record[HEADER.size + 100 : HEADER.size + 100 + HEADER.size] = MAGIC + bytes(HEADER.size - 4)
    records = parser.feed(bytes(record))
    assert [r.pcm for r in records] == [bytes(record[HEADER.size :])]
    assert not records[0].discontinuity


def test_audio_ending_like_the_magic_is_not_held_back() -> None:
    """A record whose audio ends in the start of the magic comes out right away."""
    writer = _Writer()
    parser = _parser()
    record = bytearray(writer.record(FLAG_TIMED, 352, SECOND_NS))
    record[-3:] = MAGIC[:3]
    assert parser.feed(bytes(record)) == [
        PipeRecord(FLAG_TIMED, SECOND_NS, bytes(record[HEADER.size :]))
    ]


async def _collect(gen: object, count: int) -> list[bytes]:
    return [await anext(gen) for _ in range(count)]  # type: ignore[call-overload]


def _reader(tmp_path: Path) -> tuple[BlockAudioReader, str]:
    path = str(tmp_path / "audio")
    os.mkfifo(path)
    return BlockAudioReader(path, RATE, CHANNELS, BIT_DEPTH, logging.getLogger(__name__)), path


async def test_reader_times_the_stream(tmp_path: Path) -> None:
    """The reader drops the lead-in, times gaps by continuation and flushes on FLUSH."""
    reader, path = _reader(tmp_path)
    await reader.start()
    timing = LiveTiming(RATE)
    stream = reader.iter_audio(timing)
    first = asyncio.ensure_future(_collect(stream, 2))
    await asyncio.sleep(0)
    writer = _Writer()
    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(
            fd,
            writer.record(0, 100, fill=0)  # lead-in
            + writer.record(FLAG_TIMED, 441, SECOND_NS)
            + writer.record(0, 441, fill=0),  # missing packets after timed audio
        )
        assert await asyncio.wait_for(first, 2) == [b"\x01" * 1764, bytes(1764)]
        # the missing packets continue the timed audio
        assert timing.play_time(441, 441, RATE) == (0, SECOND_NS + 10_000_000)

        os.write(
            fd, writer.record(FLAG_FLUSH) + writer.record(FLAG_TIMED, 441, 5 * SECOND_NS, fill=3)
        )
        assert await asyncio.wait_for(anext(stream), 2) == b"\x03" * 1764
        assert timing.epoch == 1
        assert timing.play_time(0, 1323, RATE) == (882, 5 * SECOND_NS)
    finally:
        os.close(fd)
        await reader.stop()
        await stream.aclose()


async def test_lost_records_restart_the_timeline(tmp_path: Path) -> None:
    """After lost records, what was queued is dropped and the next audio starts anew."""
    reader, path = _reader(tmp_path)
    await reader.start()
    timing = LiveTiming(RATE)
    stream = reader.iter_audio(timing)
    writer = _Writer()
    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(fd, writer.record(FLAG_TIMED, 441, SECOND_NS))
        assert await asyncio.wait_for(anext(stream), 2) == b"\x01" * 1764
        # a record cut short (e.g. shairport-sync's player cancelled mid-record), and the
        # stop marker that went with it
        cut = writer.record(FLAG_TIMED, 441, SECOND_NS + 10_000_000)[:100]
        writer.record(FLAG_STOP)
        os.write(fd, cut + writer.record(FLAG_TIMED, 441, 7 * SECOND_NS, fill=4))
        assert await asyncio.wait_for(anext(stream), 2) == b"\x04" * 1764
        assert timing.epoch == 1
        assert timing.play_time(441, 441, RATE) == (0, 7 * SECOND_NS)
    finally:
        os.close(fd)
        await reader.stop()
        await stream.aclose()


async def test_new_stream_replaces_the_previous_one(tmp_path: Path) -> None:
    """A second stream takes over and the first one ends."""
    reader, _ = _reader(tmp_path)
    await reader.start()
    first = reader.iter_audio(LiveTiming(RATE))
    pending = asyncio.ensure_future(anext(first))
    await asyncio.sleep(0)
    second = reader.iter_audio(LiveTiming(RATE))
    second_pending = asyncio.ensure_future(anext(second))
    await asyncio.sleep(0)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, 2)
    second_pending.cancel()
    await reader.stop()


async def test_reader_keeps_up_with_a_long_stream(tmp_path: Path) -> None:
    """A stream that keeps reading never loses audio, however long it runs."""
    reader, path = _reader(tmp_path)
    await reader.start()
    stream = reader.iter_audio(LiveTiming(RATE))
    writer = _Writer()
    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    frames = RATE // 100
    try:
        for index in range((MAX_QUEUED_SECONDS + 2) * 100):
            record = writer.record(
                FLAG_TIMED, frames, SECOND_NS + index * 10_000_000, fill=index % 256
            )
            assert os.write(fd, record) == len(record)
            chunk = await asyncio.wait_for(anext(stream), 2)
            assert chunk == bytes([index % 256]) * frames * BYTES_PER_FRAME
    finally:
        os.close(fd)
        await reader.stop()
        await stream.aclose()
