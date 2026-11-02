"""Tests for following a live source's play times on one continuous timeline."""

from __future__ import annotations

import asyncio
from itertools import pairwise
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pytest
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.push_stream import MAIN_CHANNEL, PushStream
from aiosendspin.server.roles import AudioRequirements

from music_assistant.providers.sendspin.live_timeline import (
    MAX_DEVIATION_US,
    MIN_LEAD_US,
    DriftResampler,
    LiveTimeline,
    Placement,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_RATE = 48_000
_FRAMES = 4800  # one 100 ms producer slice
_START_US = 10_000_000
_F32 = AudioFormat(sample_rate=_RATE, bit_depth=32, channels=2, sample_type="float")


def _run(
    timeline: LiveTimeline,
    play_us: Sequence[float],
    *,
    now_us: Sequence[int] | None = None,
    floor_us: int = 0,
) -> list[Placement]:
    """
    Place chunks at the given play times and commit them like aiosendspin does.

    A continuing commit due sooner than ``floor_us`` from now is moved to that floor, as
    aiosendspin does for its players' buffer. The resampler is modelled as putting out
    ``frames * (1 + adjust)`` frames, without delay.
    """
    placements: list[Placement] = []
    cursor = 0.0
    produced = 0.0
    for i, play in enumerate(play_us):
        now = now_us[i] if now_us is not None else 0
        placement = timeline.place(play, _FRAMES, now)
        assert placement is not None
        placements.append(placement)
        frames_in = _FRAMES - placement.skip_frames
        if not frames_in:
            continue
        if placement.play_start_us is not None:
            cursor, produced = placement.play_start_us, 0.0
        else:
            cursor = max(cursor, now + floor_us)
        produced += frames_in * (1 + placement.adjust)
        frames_out = round(produced)
        produced -= frames_out
        timeline.committed(placement, round(cursor), frames_in, frames_out)
        cursor += frames_out * 1e6 / _RATE
    return placements


def _deviation_us(timeline: LiveTimeline, play: float) -> float:
    """Return how far the next chunk's play time is from the end of the timeline."""
    assert timeline._end_us is not None
    return play - timeline._end_us


def _drifting(ppm: float, chunks: int, jitter_us: float = 0) -> list[float]:
    """Return play times of a source whose clock runs ``ppm`` fast, with uniform jitter."""
    jitter = np.random.default_rng(0).uniform(-jitter_us, jitter_us, chunks)
    step = 100_000 * (1 + ppm / 1e6)
    return [_START_US + i * step + float(jitter[i]) for i in range(chunks)]


@pytest.mark.parametrize("ppm", [14, -14, 300])
def test_drift_is_absorbed_without_cuts(ppm: float) -> None:
    """A drifting source stays on one timeline, kept within a millisecond after settling."""
    timeline = LiveTimeline(_RATE)
    chunks = 6000  # 10 minutes
    play = _drifting(ppm, chunks + 1)

    placements = _run(timeline, play[:chunks])

    assert [p.play_start_us for p in placements].count(None) == chunks - 1
    assert timeline.skipped_frames == 0
    assert abs(_deviation_us(timeline, play[chunks])) < 1_000
    assert placements[-1].adjust * 1e6 == pytest.approx(ppm, abs=5)


def test_jitter_is_not_mistaken_for_drift() -> None:
    """Play times jittering by 2 ms neither break the timeline nor swing the ratio."""
    timeline = LiveTimeline(_RATE)

    placements = _run(timeline, _drifting(0, 3000, jitter_us=2_000))

    assert timeline.anchors == 1
    assert timeline.skipped_frames == 0
    # a quarter of the range is ~0.2 cent of pitch, far below what is audible
    assert max(abs(p.adjust) for p in placements) < 125e-6


@pytest.mark.parametrize(
    ("jump_us", "explicit_starts", "skipped_frames"),
    [
        pytest.param(30_000, [_START_US, _START_US + 230_000], 0, id="gap_reanchors"),
        pytest.param(-30_000, [_START_US], 1440, id="repeat_skips_head"),
        pytest.param(MAX_DEVIATION_US - 1_000, [_START_US], 0, id="small_jump_is_steered"),
    ],
)
def test_jumps_in_play_times(jump_us: int, explicit_starts: list[int], skipped_frames: int) -> None:
    """A break forward anchors anew, one backward skips what was already committed."""
    timeline = LiveTimeline(_RATE)
    play = [_START_US + i * 100_000 + (jump_us if i >= 2 else 0) for i in range(4)]

    placements = _run(timeline, play)

    assert [p.play_start_us for p in placements if p.play_start_us] == explicit_starts
    assert timeline.skipped_frames == skipped_frames


def test_reset_anchors_anew_and_ignores_the_commit_in_flight() -> None:
    """A commit that completes after a reset does not bring the old timeline back."""
    timeline = LiveTimeline(_RATE)
    _run(timeline, [_START_US])
    in_flight = timeline.place(_START_US + 100_000, _FRAMES, 0)
    assert in_flight is not None

    timeline.reset()
    timeline.committed(in_flight, _START_US + 100_000, _FRAMES, _FRAMES)

    placement = timeline.place(_START_US + 50_000, _FRAMES, 0)
    assert placement is not None
    assert placement.play_start_us == _START_US + 50_000


def _leads(chunks: int, short_until: int, short_us: int, long_us: int) -> list[int]:
    """
    Return the times at which a source sends chunks that play 100 ms apart.

    They go out ``short_us`` ahead of their play times until chunk ``short_until``, then the
    source catches up to ``long_us`` ahead within a second.
    """
    leads = [short_us if i < short_until else long_us for i in range(chunks)]
    for i in range(short_until, min(chunks, short_until + 10)):
        leads[i] = short_us + (long_us - short_us) * (i - short_until + 1) // 10
    return [_START_US + i * 100_000 - lead for i, lead in enumerate(leads)]


def test_short_lead_is_won_back_once_the_source_sends_further_ahead() -> None:
    """
    Audio sent too late for the players' buffer plays late until the source sends further ahead.

    Then the audio that plays too late is skipped in one cut, back onto the source's play times.
    """
    timeline = LiveTimeline(_RATE)
    play = [_START_US + i * 100_000 for i in range(60)]

    placements = _run(timeline, play, now_us=_leads(60, 20, 300_000, 1_200_000), floor_us=600_000)

    assert timeline.rebases == 1
    assert timeline.latency_us == 0
    skipped = [i for i, p in enumerate(placements) if p.skip_frames]
    # one cut: the 300 ms the players were behind, in consecutive chunks
    assert skipped == list(range(skipped[0], skipped[0] + len(skipped)))
    assert timeline.skipped_frames == pytest.approx(0.3 * _RATE, abs=_RATE // 1000)
    assert abs(_deviation_us(timeline, _START_US + 6_000_000)) < 1_000


def test_lead_that_stays_short_keeps_the_latency() -> None:
    """While the source's lead stays short, no audio is skipped to win the latency back."""
    timeline = LiveTimeline(_RATE)
    play = [_START_US + i * 100_000 for i in range(60)]

    _run(timeline, play, now_us=_leads(60, 60, 300_000, 300_000), floor_us=600_000)

    assert timeline.rebases == 1
    assert timeline.latency_us == pytest.approx(300_000, abs=1_000)
    assert timeline.skipped_frames == 0


@pytest.mark.parametrize(
    ("lead_us", "latency_us"),
    [
        pytest.param(1_200_000, 0, id="enough_lead_drops_the_old_latency"),
        pytest.param(400_000, 220_000, id="short_lead_takes_what_it_needs"),
    ],
)
def test_new_anchor_takes_only_the_latency_its_lead_needs(lead_us: int, latency_us: int) -> None:
    """After a flush, the latency left from before neither carries over nor gets moved again."""
    timeline = LiveTimeline(_RATE)
    play = [_START_US + i * 100_000 for i in range(5)]
    _run(timeline, play, now_us=[p - 300_000 for p in play], floor_us=600_000)
    assert timeline.latency_us == pytest.approx(300_000, abs=1_000)

    timeline.reset()
    play_us = _START_US + 10_000_000
    anchor = timeline.place(play_us, _FRAMES, play_us - lead_us)

    assert anchor is not None
    assert timeline.latency_us == pytest.approx(latency_us, abs=1_000)
    assert anchor.play_start_us == pytest.approx(play_us + latency_us, abs=1_000)


@pytest.mark.parametrize(
    ("play_us", "end_us"),
    [
        pytest.param(_START_US, _START_US + 100_001, id="new_anchor"),
        pytest.param(_START_US + 100_000, _START_US + 200_001, id="continuing"),
    ],
)
def test_late_audio_is_dropped_and_anchors_anew(play_us: int, end_us: int) -> None:
    """Audio that would end before it can play is dropped, and the next chunk anchors."""
    timeline = LiveTimeline(_RATE)
    if play_us != _START_US:
        _run(timeline, [_START_US])

    assert timeline.place(play_us, _FRAMES, end_us - MIN_LEAD_US) is None
    assert timeline.late_frames == _FRAMES
    later = timeline.place(play_us + 500_000, _FRAMES, 0)
    assert later is not None
    assert later.play_start_us == play_us + 500_000


def _indexed(frames: int, first: int = 0) -> bytes:
    """Return stereo float32 PCM whose left channel ramps with the frame index."""
    ramp = np.arange(first, first + frames, dtype=np.float32) / 1e6
    return np.column_stack([ramp, np.zeros(frames, np.float32)]).tobytes()


@pytest.mark.parametrize("adjust", [0.0, 500e-6, -500e-6])
def test_resampler_keeps_time_alignment(adjust: float) -> None:
    """Resampled audio starts where the input did: the delay only holds frames back."""
    resampler = DriftResampler(_RATE, 2)
    channel = uuid4()
    pcm = bytearray(_FRAMES * 8 * 3)
    impulse = 1000
    pcm[impulse * 8 : impulse * 8 + 4] = np.float32(1).tobytes()

    out = b"".join(
        resampler.process(adjust, [(channel, bytes(pcm[i : i + _FRAMES * 8]))])[0][1]
        for i in range(0, len(pcm), _FRAMES * 8)
    )

    left = np.frombuffer(out, np.float32)[::2]
    # soxr's variable-rate filter lags by a constant ~2 frames (42 us)
    assert abs(int(np.argmax(left)) - impulse * (1 + adjust)) <= 3


def test_late_channel_stays_in_lock_step() -> None:
    """A channel that joins mid-stream stays within two frames (42 us) of the reference."""
    resampler = DriftResampler(_RATE, 2)
    main, member = uuid4(), uuid4()
    resampler.process(300e-6, [(main, _indexed(_FRAMES))])
    outputs = [
        resampler.process(300e-6 * (-1) ** i, [(main, chunk), (member, chunk)])
        for i, chunk in ((i, _indexed(_FRAMES, i * _FRAMES)) for i in range(1, 50))
    ]

    main_ramp, member_ramp = (
        np.frombuffer(b"".join(out[n][1] for out in outputs), np.float32)[::2] for n in (0, 1)
    )
    frames = min(len(main_ramp), len(member_ramp))
    # the member's first chunk has silence where the reference has its held-back tail
    lag_frames = 1e6 * (main_ramp[_FRAMES:frames] - member_ramp[_FRAMES:frames])
    assert np.abs(lag_frames).max() < 2


class _Stub(SimpleNamespace):
    """Has the given attributes, and a no-op for any other hook aiosendspin calls."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *_args, **_kwargs: None


class _Connection(_Stub):
    """Records the audio a player receives."""

    buffer_tracker: Any = None

    def __init__(self) -> None:
        super().__init__()
        self.received: list[tuple[int, int]] = []

    def send_binary(
        self,
        data: bytes,
        *,
        timestamp_us: int,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
        **_kwargs: object,
    ) -> bool:
        self.received.append((timestamp_us, len(data) // 4))
        if buffer_end_time_us is not None and buffer_byte_count:
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


def _pcm_player(clock: ManualClock, min_buffer_ms: int) -> tuple[PushStream, _Connection]:
    """Return a real push stream with one 16-bit PCM player."""
    loop = asyncio.get_running_loop()
    group: Any = _Stub(
        clients=[],
        group_name="group",
        has_active_stream=False,
        transformer_pool=TransformerPool(),
        _push_stream=None,
        get_channel_for_player=lambda _player_id: MAIN_CHANNEL,
    )
    server: Any = _Stub(loop=loop, clock=clock, id="server", name="server")
    client = SendspinClient(server, client_id="player")
    client._group = group
    group.clients.append(client)
    connection = _Connection()
    client.attach_connection(
        connection,  # type: ignore[arg-type]
        client_info=SimpleNamespace(  # type: ignore[arg-type]
            client_id="player",
            name="player",
            device_info=None,
            artwork_support=None,
            visualizer_support=None,
            player_support=ClientHelloPlayerSupport(
                supported_formats=[
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM, channels=2, sample_rate=_RATE, bit_depth=16
                    )
                ],
                buffer_capacity=10_000_000,
                supported_commands=[PlayerCommand.VOLUME],
            ),
        ),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    role: Any = client.role("player@v1")
    role.min_buffer_ms = min_buffer_ms
    role._audio_requirements = AudioRequirements(
        sample_rate=_RATE, bit_depth=16, channels=2, transformer=None, channel_id=MAIN_CHANNEL
    )
    connection.buffer_tracker = role.get_buffer_tracker()
    stream = PushStream(loop=loop, clock=clock, group=group)
    stream.set_live_source(True)
    return stream, connection


@pytest.mark.parametrize(
    ("lead_us", "catch_up_us", "min_buffer_ms", "gaps", "max_latency_us", "skipped_ms"),
    [
        pytest.param(1_200_000, 0, 500, 0, 0, 0, id="lead_above_the_players_buffer"),
        # the one move that gives the player its buffer, by the jitter that crossed it
        pytest.param(1_000_000, 0, 1000, 1, 10_000, 0, id="lead_below_the_players_buffer"),
        # moved by the 200 ms its lead fell short, then won back by skipping that audio,
        # which leaves the player no gap
        pytest.param(300_000, 900_000, 500, 1, 0, 200, id="short_lead_that_grows"),
    ],
)
async def test_player_receives_one_continuous_stream(
    lead_us: int,
    catch_up_us: int,
    min_buffer_ms: int,
    gaps: int,
    max_latency_us: int,
    skipped_ms: int,
) -> None:
    """A drifting, jittering source reaches a real player without gaps or overlaps."""
    clock = ManualClock(now_us_value=_START_US)
    stream, connection = _pcm_player(clock, min_buffer_ms)
    timeline = LiveTimeline(_RATE)
    resampler = DriftResampler(_RATE, 2)
    pcm = bytes(_FRAMES * 8)

    for i, play_us in enumerate(_drifting(300, 300, jitter_us=2_000)):
        placement = timeline.place(play_us + lead_us, _FRAMES, clock.now_us())
        assert placement is not None
        if placement.play_start_us is not None:
            resampler.reset()
        start = placement.skip_frames * 8
        ((_, out),) = resampler.process(placement.adjust, [(MAIN_CHANNEL, pcm[start:])])
        if out:
            stream.prepare_audio(out, _F32)
            start_us = await stream.commit_audio(play_start_us=placement.play_start_us)
            timeline.committed(placement, start_us, _FRAMES - placement.skip_frames, len(out) // 8)
        # from chunk 20 on, the source sends its chunks faster until it is catch_up_us further
        # ahead
        clock.advance_us(100_000 - (catch_up_us // 10 if 20 <= i < 30 else 0))
    stream.stop()

    spans = [(start, start + frames * 1e6 / _RATE) for start, frames in connection.received]
    assert sum(abs(nxt - end) > 1 for (_, end), (nxt, _) in pairwise(spans)) == gaps
    assert timeline.latency_us <= max_latency_us
    assert timeline.skipped_frames * 1000 // _RATE == pytest.approx(skipped_ms, abs=2)
