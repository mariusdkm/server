"""Tests for the reader of shairport-sync's audio pipe with timing headers."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

import pytest

from music_assistant.helpers.live_timing import LiveTiming
from music_assistant.providers.airplay_receiver.timed_pipe import (
    FLAG_FLUSH,
    FLAG_STOP,
    FLAG_TIMED,
    HEADER,
    HEADER_MAGIC,
    MAX_QUEUED_SECONDS,
    RAW_DETECT_BYTES,
    PipeRecord,
    TimedPipeParser,
    TimedPipeReader,
)

if TYPE_CHECKING:
    from pathlib import Path

RATE = 44100
BYTES_PER_FRAME = 4
SECOND_NS = 1_000_000_000


def _record(flags: int, frames: int = 0, play_ns: int = 0, fill: int = 1) -> bytes:
    header = HEADER.pack(HEADER_MAGIC, 1, flags, frames, 1234, play_ns, RATE, BYTES_PER_FRAME)
    return header + bytes([fill]) * frames * BYTES_PER_FRAME


STREAM = (
    _record(0, 10, fill=0)
    + _record(FLAG_TIMED, 352, SECOND_NS)
    + _record(FLAG_FLUSH)
    + _record(FLAG_TIMED, 352, 2 * SECOND_NS, fill=2)
    + _record(FLAG_STOP)
)
EXPECTED = [
    PipeRecord(0, 0, bytes(40)),
    PipeRecord(FLAG_TIMED, SECOND_NS, b"\x01" * 1408),
    PipeRecord(FLAG_FLUSH, 0, b""),
    PipeRecord(FLAG_TIMED, 2 * SECOND_NS, b"\x02" * 1408),
    PipeRecord(FLAG_STOP, 0, b""),
]


@pytest.mark.parametrize("split", range(0, len(STREAM), 7))
def test_records_split_across_reads(split: int) -> None:
    """Records come out whole wherever the reads split them."""
    parser = TimedPipeParser(RATE, BYTES_PER_FRAME)
    records = parser.feed(STREAM[:split]) + parser.feed(STREAM[split:])
    assert records == EXPECTED
    assert parser.timed
    assert parser.resyncs == 0


def test_resync_after_garbage() -> None:
    """Bytes that are no record are skipped up to the next valid header."""
    parser = TimedPipeParser(RATE, BYTES_PER_FRAME)
    parser.feed(_record(FLAG_TIMED, 10, SECOND_NS))
    records = parser.feed(b"\x00" * 100 + _record(FLAG_TIMED, 352, SECOND_NS))
    assert records == [PipeRecord(FLAG_TIMED, SECOND_NS, b"\x01" * 1408)]
    assert parser.resyncs == 1


def test_header_with_another_format_is_rejected() -> None:
    """A header announcing an unexpected output format does not count as a record."""
    parser = TimedPipeParser(48000, BYTES_PER_FRAME)
    assert parser.feed(_record(FLAG_TIMED, 352, SECOND_NS)) == []
    assert parser.timed is None


def test_plain_pcm_falls_back_to_raw() -> None:
    """Without any header (older shairport-sync), all bytes pass as untimed audio."""
    parser = TimedPipeParser(RATE, BYTES_PER_FRAME)
    pcm = bytes(range(256)) * (RAW_DETECT_BYTES // 256)
    assert parser.feed(pcm) == [PipeRecord(0, 0, pcm)]
    assert parser.timed is False
    assert parser.feed(b"more") == [PipeRecord(0, 0, b"more")]


async def _collect(gen: object, count: int) -> list[bytes]:
    return [await anext(gen) for _ in range(count)]  # type: ignore[call-overload]


async def test_reader_times_the_stream(tmp_path: Path) -> None:
    """The reader drops the lead-in, times gaps by continuation and flushes on FLUSH."""
    path = str(tmp_path / "audio")
    os.mkfifo(path)
    reader = TimedPipeReader(path, RATE, BYTES_PER_FRAME, logging.getLogger(__name__))
    await reader.start()
    timing = LiveTiming(RATE)
    stream = reader.iter_audio(timing)
    first = asyncio.ensure_future(_collect(stream, 2))
    await asyncio.sleep(0)
    writer = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(
            writer,
            _record(0, 100, fill=0)  # lead-in
            + _record(FLAG_TIMED, 441, SECOND_NS)
            + _record(0, 441, fill=0),  # missing packets after timed audio
        )
        assert await asyncio.wait_for(first, 2) == [b"\x01" * 1764, bytes(1764)]
        assert reader.timed
        # the missing packets continue the timed audio
        assert timing.play_time(441, 441, RATE) == (0, SECOND_NS + 10_000_000)

        os.write(writer, _record(FLAG_FLUSH) + _record(FLAG_TIMED, 441, 5 * SECOND_NS, fill=3))
        assert await asyncio.wait_for(anext(stream), 2) == b"\x03" * 1764
        assert timing.epoch == 1
        assert timing.play_time(0, 1323, RATE) == (882, 5 * SECOND_NS)
    finally:
        os.close(writer)
        await reader.stop()
        await stream.aclose()


async def test_new_stream_replaces_the_previous_one(tmp_path: Path) -> None:
    """A second stream takes over and the first one ends."""
    path = str(tmp_path / "audio")
    os.mkfifo(path)
    reader = TimedPipeReader(path, RATE, BYTES_PER_FRAME, logging.getLogger(__name__))
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
    path = str(tmp_path / "audio")
    os.mkfifo(path)
    reader = TimedPipeReader(path, RATE, BYTES_PER_FRAME, logging.getLogger(__name__))
    await reader.start()
    stream = reader.iter_audio(LiveTiming(RATE))
    writer = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    frames = RATE // 100
    try:
        for index in range((MAX_QUEUED_SECONDS + 2) * 100):
            record = _record(FLAG_TIMED, frames, SECOND_NS + index * 10_000_000, fill=index % 256)
            assert os.write(writer, record) == len(record)
            chunk = await asyncio.wait_for(anext(stream), 2)
            assert chunk == bytes([index % 256]) * frames * BYTES_PER_FRAME
    finally:
        os.close(writer)
        await reader.stop()
        await stream.aclose()
