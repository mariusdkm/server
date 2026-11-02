"""Tests for the play times of live PCM streams."""

from __future__ import annotations

import asyncio

import pytest

from music_assistant.helpers.live_timing import LiveTiming

SECOND_NS = 1_000_000_000


def _timing() -> LiveTiming:
    """Two records of 100 frames at 1 kHz: one at t=10 s, one at t=20 s."""
    timing = LiveTiming(1000)
    timing.add_audio(100, 10 * SECOND_NS)
    timing.add_audio(100, 20 * SECOND_NS)
    return timing


@pytest.mark.parametrize(
    ("start", "frames", "sample_rate", "expected"),
    [
        pytest.param(50, 100, 1000, (0, 10_050_000_000), id="inside_a_record"),
        pytest.param(100, 200, 2000, (0, 10_050_000_000), id="consumer_rate"),
        pytest.param(150, 100, 1000, (0, 20_050_000_000), id="later_record"),
        pytest.param(200, 100, 1000, None, id="beyond_the_known_audio"),
    ],
)
def test_play_time(
    start: int, frames: int, sample_rate: int, expected: tuple[int, int] | None
) -> None:
    """The play time of a range is that of its first frame."""
    assert _timing().play_time(start, frames, sample_rate) == expected


def test_silence_is_untimed_and_shifts_later_audio() -> None:
    """Inserted silence has no play time and moves the following audio back."""
    timing = LiveTiming(1000)
    timing.add_audio(100, 10 * SECOND_NS)
    timing.add_silence(100, 30)
    timing.add_audio(100, 20 * SECOND_NS)
    assert timing.play_time(100, 130, 1000) == (30, 20 * SECOND_NS)


def test_silence_reported_after_later_audio_was_added() -> None:
    """Silence inserted before audio the source already yielded lands in the right place."""
    timing = _timing()
    timing.add_silence(100, 10)
    timing.add_silence(100, 10)
    assert timing.play_time(100, 120, 1000) == (20, 20 * SECOND_NS)


def test_new_epoch_marks_earlier_audio_stale() -> None:
    """Audio added before a flush has no play time any more."""
    timing = LiveTiming(1000)
    timing.add_audio(100, 10 * SECOND_NS)
    timing.new_epoch()
    timing.add_audio(100, 30 * SECOND_NS)
    assert timing.play_time(0, 200, 1000) == (100, 30 * SECOND_NS)


async def test_wait_for_new_epoch() -> None:
    """Waiters wake up on a flush and learn the new epoch."""
    timing = LiveTiming(1000)
    waiter = asyncio.create_task(timing.wait_for_new_epoch(0))
    await asyncio.sleep(0)
    assert not waiter.done()
    timing.new_epoch()
    assert await waiter == 1


def test_prune_forgets_passed_audio() -> None:
    """Pruned ranges have no play time any more, later ones keep theirs."""
    timing = _timing()
    timing.prune(100, 1000)
    assert timing.play_time(0, 200, 1000) == (100, 20 * SECOND_NS)


def test_history_is_bounded_without_a_consumer_pruning() -> None:
    """Without prune calls, only the most recent stretch of play times is kept."""
    timing = LiveTiming(1000)
    # an hour of 10 ms records and periodic silence, as a long session with no timed consumer
    for i in range(360_000):
        timing.add_audio(10, i * 10_000_000)
        if i % 1000 == 0:
            timing.add_silence((i + 1) * 10, 50)
    assert len(timing._spans) <= 31 * 100 + 31
    end = 360_000 * 10 + 360 * 50
    # the newest audio keeps its play time, audio long gone has none
    assert timing.play_time(end - 10, 10, 1000) == (0, 3_599_990_000_000)
    assert timing.play_time(0, 10, 1000) is None
