"""Tests for Sendspin playing a live source at the play times the source provides."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiosendspin.server.audio import AudioFormat as SendspinAudioFormat
from music_assistant_models.dsp import DSPConfig
from music_assistant_models.enums import MediaType
from music_assistant_models.media_items.audio_format import AudioFormat

from music_assistant.helpers.live_timing import LiveTiming
from music_assistant.models.player import PlayerMedia
from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.sendspin.playback import SendspinPlaybackSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

_SAMPLE_RATE = 48_000
_FRAME_SIZE = 4 * 2
_CHUNK_FRAMES = 4800  # one 100 ms producer slice
_CHUNK = b"\x00" * (_CHUNK_FRAMES * _FRAME_SIZE)
_PCM_FORMAT = AudioFormat(sample_rate=_SAMPLE_RATE, bit_depth=32, channels=2)
_SENDSPIN_PCM_FORMAT = SendspinAudioFormat(
    sample_rate=_SAMPLE_RATE, bit_depth=32, channels=2, sample_type="float"
)
_NOW_US = 50_000_000
_PLAY_US = _NOW_US + 1_000_000
_PLAY_NS = _PLAY_US * 1000
# frames a new anchor's resampler holds back
_HELD = 240


class _FakePushStream:
    """Stands in for aiosendspin's PushStream and records what gets committed."""

    def __init__(self) -> None:
        # explicit start (None in auto mode), start used and frames of each commit
        self.commits: list[tuple[int | None, int, int]] = []
        self.clears = 0
        self.committed = asyncio.Event()
        # how far the clock moves while a commit waits for the buffer to drain
        self.commit_wait_us = 0
        # when set, each commit stays in flight until the event is set
        self.commit_gate: asyncio.Event | None = None
        self.commit_started = asyncio.Event()
        self._now_us = _NOW_US
        self._pcm_len = 0
        self._cursor_us = float(_NOW_US)

    def set_live_source(self, _live: bool) -> None:
        pass

    def prepare_audio(self, pcm: bytes, *_args: object, **_kwargs: object) -> None:
        self._pcm_len = len(pcm)

    async def commit_audio(self, *, play_start_us: int | None = None) -> int:
        # an explicit start moves the cursor, auto mode continues it
        if play_start_us is not None:
            self._cursor_us = play_start_us
        start_us = int(self._cursor_us)
        frames = self._pcm_len // _FRAME_SIZE
        self._cursor_us += frames * 1e6 / _SAMPLE_RATE
        self.commits.append((play_start_us, start_us, frames))
        self.commit_started.set()
        if self.commit_gate is not None:
            await self.commit_gate.wait()
        self.committed.set()
        return start_us

    @property
    def explicit_starts(self) -> list[int | None]:
        return [explicit for explicit, _, _ in self.commits]

    async def sleep_to_limit_buffer(self, _limit_us: int) -> None:
        self._now_us += self.commit_wait_us

    def now_us(self) -> int:
        return self._now_us

    def stop(self, *, keep_stream: bool = False) -> None:
        pass

    def clear(self) -> None:
        self.clears += 1


async def _play(
    monkeypatch: pytest.MonkeyPatch,
    timing: LiveTiming,
    source: Callable[[_FakePushStream], AsyncIterator[bytes]],
    untimed_chunks: int = 0,
    media_type: MediaType = MediaType.AUDIO_SOURCE,
    push_stream: _FakePushStream | None = None,
) -> tuple[_FakePushStream, MagicMock]:
    """
    Run one playback session of a live source that provides the given timing.

    :param untimed_chunks: Chunks committed before the source knows its timing.
    :param media_type: Media type the session plays.
    :param push_stream: Push stream to use, for tests that set it up beforehand.
    """
    player = MagicMock()
    player.player_id = "leader"
    player.mass.config.get_player_dsp_config.return_value = DSPConfig(enabled=False)
    player.mass.config.get_raw_player_config_value.return_value = "stereo"
    player.api.group.stop = AsyncMock()
    provider = MagicMock(spec=PluginProvider)
    provider.get_audio_source_timing.side_effect = [None] * untimed_chunks + [timing] * 100
    player.mass.get_provider.return_value = provider
    push_stream = push_stream or _FakePushStream()
    player.mass.streams.get_stream.return_value = source(push_stream)

    session = SendspinPlaybackSession(player)
    monkeypatch.setattr(
        session, "_select_session_pcm_formats", lambda: (_PCM_FORMAT, _SENDSPIN_PCM_FORMAT)
    )
    monkeypatch.setattr(session, "_create_push_stream", lambda: push_stream)
    monkeypatch.setattr(session, "_wait_for_buffer_drain", AsyncMock())
    monkeypatch.setattr(
        "music_assistant.providers.sendspin.playback.import_module_in_thread", AsyncMock()
    )
    await session._run_playback(
        PlayerMedia(
            uri="airplay://source",
            media_type=media_type,
            source_id="leader",
            queue_item_id="item",
        )
    )
    return push_stream, provider


async def _play_chunks(
    monkeypatch: pytest.MonkeyPatch, timing: LiveTiming, chunks: int
) -> _FakePushStream:
    async def _source(_push_stream: _FakePushStream) -> AsyncIterator[bytes]:
        for _ in range(chunks):
            yield _CHUNK

    push_stream, _ = await _play(monkeypatch, timing, _source)
    return push_stream


async def test_commits_at_the_sources_play_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence is dropped, the audio anchors at its play time and then continues in auto mode."""
    timing = LiveTiming(_SAMPLE_RATE)
    timing.add_silence(0, _CHUNK_FRAMES)
    for i in range(3):
        timing.add_audio(_CHUNK_FRAMES, _PLAY_NS + i * 100_000_000)

    push_stream = await _play_chunks(monkeypatch, timing, 4)

    assert push_stream.explicit_starts == [_PLAY_US, None, None]
    assert push_stream.commits[0][2] == _CHUNK_FRAMES - _HELD


@pytest.mark.parametrize(
    ("jump_us", "explicit_starts", "second_frames"),
    [
        pytest.param(1_500, [_PLAY_US, None], _CHUNK_FRAMES, id="offset_is_steered"),
        pytest.param(30_000, [_PLAY_US, _PLAY_US + 130_000], _CHUNK_FRAMES - _HELD, id="gap"),
        # the 240 frames the resampler holds back are already part of the timeline
        pytest.param(-30_000, [_PLAY_US, None], _CHUNK_FRAMES - 1440, id="overlap_is_skipped"),
    ],
)
async def test_breaks_in_the_play_times(
    monkeypatch: pytest.MonkeyPatch,
    jump_us: int,
    explicit_starts: list[int | None],
    second_frames: int,
) -> None:
    """Only a jump forward is committed at an explicit time, never one into committed audio."""
    timing = LiveTiming(_SAMPLE_RATE)
    timing.add_audio(_CHUNK_FRAMES, _PLAY_NS)
    timing.add_audio(_CHUNK_FRAMES, (_PLAY_US + 100_000 + jump_us) * 1000)

    push_stream = await _play_chunks(monkeypatch, timing, 2)

    assert push_stream.explicit_starts == explicit_starts
    assert push_stream.commits[1][2] == pytest.approx(second_frames, abs=1)


async def test_late_audio_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audio due before it can reach the clients is dropped, later audio anchors anew."""
    timing = LiveTiming(_SAMPLE_RATE)
    timing.add_audio(_CHUNK_FRAMES, (_NOW_US - 500_000) * 1000)
    timing.add_audio(_CHUNK_FRAMES, _PLAY_NS)

    push_stream = await _play_chunks(monkeypatch, timing, 2)

    assert push_stream.explicit_starts == [_PLAY_US]


async def test_flush_clears_the_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flush clears what the clients hold and drops audio from before it."""
    timing = LiveTiming(_SAMPLE_RATE)
    timing.add_audio(_CHUNK_FRAMES, _PLAY_NS)
    timing.add_audio(_CHUNK_FRAMES, _PLAY_NS + 100_000_000)

    async def _source(push_stream: _FakePushStream) -> AsyncIterator[bytes]:
        yield _CHUNK
        await asyncio.wait_for(push_stream.committed.wait(), 2)
        timing.new_epoch()
        await asyncio.sleep(0)
        yield _CHUNK

    push_stream, _ = await _play(monkeypatch, timing, _source)

    assert push_stream.explicit_starts == [_PLAY_US]
    assert push_stream.clears == 1


async def test_timing_found_after_the_first_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence sent before the source knows its timing does not keep the session untimed."""
    timing = LiveTiming(_SAMPLE_RATE)
    timing.add_silence(0, _CHUNK_FRAMES)
    timing.add_audio(_CHUNK_FRAMES, _PLAY_NS)

    async def _source(_push_stream: _FakePushStream) -> AsyncIterator[bytes]:
        for _ in range(2):
            yield _CHUNK

    push_stream, _ = await _play(monkeypatch, timing, _source, untimed_chunks=1)

    assert push_stream.explicit_starts == [None, _PLAY_US]
    # the silence went out on Sendspin's own timeline, and must not overlap the play times
    assert push_stream.clears == 1


async def test_queue_track_does_not_look_for_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only live AudioSources can carry play times, so other media skip the lookup."""

    async def _source(_push_stream: _FakePushStream) -> AsyncIterator[bytes]:
        for _ in range(3):
            yield _CHUNK

    push_stream, provider = await _play(
        monkeypatch, LiveTiming(_SAMPLE_RATE), _source, media_type=MediaType.TRACK
    )

    provider.get_audio_source_timing.assert_not_called()
    assert push_stream.explicit_starts == [None] * 3
    assert {frames for _, _, frames in push_stream.commits} == {_CHUNK_FRAMES}


async def test_flush_during_the_first_commit_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flush while the first timed commit is still in flight clears the clients."""
    timing = LiveTiming(_SAMPLE_RATE)
    timing.add_audio(_CHUNK_FRAMES, _PLAY_NS)
    push_stream = _FakePushStream()
    push_stream.commit_gate = asyncio.Event()

    async def _source(push_stream: _FakePushStream) -> AsyncIterator[bytes]:
        yield _CHUNK
        await asyncio.wait_for(push_stream.commit_started.wait(), 2)
        timing.new_epoch()
        await asyncio.sleep(0)
        assert push_stream.commit_gate is not None
        push_stream.commit_gate.set()

    push_stream, _ = await _play(monkeypatch, timing, _source, push_stream=push_stream)

    assert push_stream.explicit_starts == [_PLAY_US]
    assert push_stream.clears == 1


async def test_flush_anchors_the_next_audio_anew(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a flush the next audio plays at its own time, even before the old timeline end."""
    timing = LiveTiming(_SAMPLE_RATE)
    timing.add_audio(_CHUNK_FRAMES, _PLAY_NS)

    async def _source(push_stream: _FakePushStream) -> AsyncIterator[bytes]:
        yield _CHUNK
        await asyncio.wait_for(push_stream.committed.wait(), 2)
        timing.new_epoch()
        await asyncio.sleep(0)
        timing.add_audio(_CHUNK_FRAMES, _PLAY_NS + 50_000_000)
        yield _CHUNK

    push_stream, _ = await _play(monkeypatch, timing, _source)

    assert push_stream.explicit_starts == [_PLAY_US, _PLAY_US + 50_000]
    assert push_stream.clears == 1
