"""Place a live source on the Sendspin timeline at the times the source wants it played."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import soxr

if TYPE_CHECKING:
    from uuid import UUID

# beyond this the source's play times broke (seek, pause, dropout) instead of drifting
MAX_DEVIATION_US = 20_000
# under one cent of pitch, and far more than the skew between two real clocks
_MAX_ADJUST = 500e-6
# AirPlay play times jitter by up to ~2 ms, so the loop acts on a smoothed deviation
_SMOOTHING_S = 2.0
# settling time of the critically damped drift loop; slow enough to stay inaudible
_LOOP_TIME_CONSTANT_S = 25.0
# aiosendspin moved a continuing commit later, as it does when audio comes too late for the
# players' minimum buffer
_REBASE_US = 1_000


@dataclass(frozen=True, slots=True)
class Placement:
    """Where a chunk goes on the timeline."""

    # leading frames that repeat audio already committed
    skip_frames: int
    # explicit start for a new anchor; None continues the committed timeline
    play_start_us: int | None
    # fraction of extra output per input frame to resample the chunk with
    adjust: float
    generation: int


class LiveTimeline:
    """
    Keep one continuous timeline that follows the play times of a live source.

    Play times anchor the timeline and reveal breaks in it. Drift between the source's clock
    and the timeline is absorbed by resampling at the ratio each placement carries, never by
    cutting the audio.
    """

    def __init__(self, sample_rate: int) -> None:
        """
        Initialize the timeline.

        :param sample_rate: Sample rate of the audio placed on the timeline.
        """
        self.sample_rate = sample_rate
        self.anchors = 0
        self.skipped_frames = 0
        self.late_frames = 0
        self.rebases = 0
        # how much later than the source asks the audio plays, so the players get their buffer
        self.latency_us = 0
        self._generation = 0
        self._end_us: float | None = None
        # input frames fed to the resampler that it has not put out yet
        self._held_frames = 0.0
        self._adjust = 0.0
        self._deviation_us = 0.0
        # the source's clock skew, learned by the loop; it survives resets like the clocks do
        self._skew = 0.0

    def place(self, play_us: float, frames: int, earliest_end_us: int) -> Placement | None:
        """
        Decide where a chunk goes, or return None when it can no longer play in time.

        :param play_us: Time at which the source wants the chunk's first frame played.
        :param frames: Number of frames in the chunk.
        :param earliest_end_us: Committed audio must end after this time to still be played.
        """
        rate = self.sample_rate
        play_us += self.latency_us
        if self._end_us is not None:
            deviation_us = play_us - self._held_frames * 1e6 / rate - self._end_us
            if abs(deviation_us) <= MAX_DEVIATION_US:
                if self._end_us + frames * 1e6 / rate < earliest_end_us:
                    self._late(frames)
                    return None
                self._steer(deviation_us, frames)
                return Placement(0, None, self._adjust, self._generation)
            if deviation_us < 0:
                skip = min(frames, math.ceil(-deviation_us * rate / 1e6))
                if self._end_us + (frames - skip) * 1e6 / rate < earliest_end_us:
                    self._late(frames)
                    return None
                self.skipped_frames += skip
                return Placement(skip, None, self._adjust, self._generation)
        if play_us + frames * 1e6 / rate < earliest_end_us:
            self._late(frames)
            return None
        self.anchors += 1
        self._deviation_us = 0.0
        self._adjust = self._clamp(self._skew)
        return Placement(0, round(play_us), self._adjust, self._generation)

    def committed(
        self, placement: Placement, start_us: int, frames_in: int, frames_out: int
    ) -> None:
        """
        Record a commit made for a placement.

        :param placement: Placement the commit was made for.
        :param start_us: Start time the commit returned.
        :param frames_in: Frames of the chunk fed to the resampler.
        :param frames_out: Frames the resampler put out and that were committed.
        """
        if placement.generation != self._generation:
            # reset while the commit was in flight
            return
        if placement.play_start_us is not None:
            self._held_frames = 0.0
        elif self._end_us is not None and (moved_us := start_us - self._end_us) > _REBASE_US:
            # keep the delay; skipping audio to win it back would only get it moved again
            self.latency_us += round(moved_us)
            self.rebases += 1
        self._held_frames += frames_in - frames_out / (1 + placement.adjust)
        self._end_us = start_us + frames_out * 1e6 / self.sample_rate

    def reset(self) -> None:
        """Forget the timeline, so the next chunk anchors anew."""
        self._generation += 1
        self._end_us = None

    def summary(self) -> str:
        """Return a report of the timeline since the last summary, and start a new one."""
        rate = self.sample_rate
        text = (
            f"resample {self._adjust * 1e6:+.0f} ppm, deviation {self._deviation_us / 1e3:+.2f} ms,"
            f" anchors {self.anchors}, skipped {self.skipped_frames * 1000 // rate} ms,"
            f" late {self.late_frames * 1000 // rate} ms, rebases {self.rebases},"
            f" latency {self.latency_us / 1e3:.1f} ms"
        )
        self.anchors = self.skipped_frames = self.late_frames = self.rebases = 0
        return text

    def _steer(self, deviation_us: float, frames: int) -> None:
        dt = frames / self.sample_rate
        self._deviation_us += (deviation_us - self._deviation_us) * min(1.0, dt / _SMOOTHING_S)
        error = self._deviation_us / 1e6
        skew = self._skew + error * dt / _LOOP_TIME_CONSTANT_S**2
        raw = 2 * error / _LOOP_TIME_CONSTANT_S + skew
        if abs(raw) < _MAX_ADJUST:
            # learning the skew while saturated would overshoot once the error is gone
            self._skew = skew
        self._adjust = self._clamp(raw)

    def _late(self, frames: int) -> None:
        self.late_frames += frames
        self._end_us = None

    @staticmethod
    def _clamp(adjust: float) -> float:
        return max(-_MAX_ADJUST, min(_MAX_ADJUST, adjust))


class DriftResampler:
    """Resample the channels of one stream in lock-step, at a ratio close to one."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        """
        Initialize the resampler.

        :param sample_rate: Sample rate of the float32 PCM in and out.
        :param channels: Number of interleaved audio channels.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._streams: dict[UUID, soxr.ResampleStream] = {}
        # input frames each stream holds back
        self._held: dict[UUID, float] = {}

    def process(self, adjust: float, chunks: list[tuple[UUID, bytes]]) -> list[tuple[UUID, bytes]]:
        """
        Resample one chunk of each channel, the first being the reference.

        :param adjust: Fraction of extra output per input frame.
        :param chunks: Channel id and float32 PCM of each channel.
        """
        rate = self.sample_rate
        frame_size = 4 * self.channels
        reference_held = round(self._held.get(chunks[0][0], 0.0))
        out: list[tuple[UUID, bytes]] = []
        for channel_id, pcm in chunks:
            stream = self._streams.get(channel_id)
            joining = stream is None
            if stream is None:
                stream = self._streams[channel_id] = soxr.ResampleStream(
                    rate * (1 + _MAX_ADJUST), rate, self.channels, dtype="float32", vr=True
                )
                self._held[channel_id] = 0.0
            stream.set_io_ratio(rate * (1 - adjust), rate)
            if joining and out:
                # a channel joining late starts as far behind as the reference, so its
                # frames line up with the reference's
                self._feed(channel_id, stream, bytes(reference_held * frame_size), adjust)
            # aiosendspin times each channel by its own frames, so a frame of rounding
            # between them needs no trimming
            out.append((channel_id, self._feed(channel_id, stream, pcm, adjust)))
        return out

    def reset(self) -> None:
        """Drop the audio held back, for a new anchor."""
        self._streams.clear()
        self._held.clear()

    def _feed(
        self, channel_id: UUID, stream: soxr.ResampleStream, pcm: bytes, adjust: float
    ) -> bytes:
        frames = np.frombuffer(pcm, dtype=np.float32).reshape(-1, self.channels)
        resampled = stream.resample_chunk(frames)
        self._held[channel_id] += len(frames) - len(resampled) / (1 + adjust)
        return np.ascontiguousarray(resampled).tobytes()
