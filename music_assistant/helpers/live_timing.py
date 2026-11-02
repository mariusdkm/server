"""Play times of a live PCM stream whose source tells when each frame should be heard."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

# play times kept behind the newest frame when no consumer prunes them (e.g. a player that
# doesn't use them); a consumer that reads further behind than this is too late to play anyway
_MAX_HISTORY_SECONDS = 30


@dataclass(slots=True)
class _Span:
    # output frames: the stream as the consumer reads it, including inserted silence
    start: int
    frames: int
    # frames of the source's own audio before this span
    source_start: int
    # CLOCK_MONOTONIC_RAW ns of the first frame; None for inserted silence
    play_ns: int | None
    epoch: int

    @property
    def end(self) -> int:
        return self.start + self.frames


class LiveTiming:
    """
    Map frame positions of one live PCM stream to the times its source wants them played.

    The source adds its audio as it yields it, the server reports the silence it inserts
    into the stream, and the consumer looks up ranges of the stream it reads. Positions
    are frames at ``sample_rate``; times are CLOCK_MONOTONIC_RAW nanoseconds. The epoch
    advances whenever the source discards audio it already delivered (flush or stop).
    """

    def __init__(self, sample_rate: int) -> None:
        """
        Initialize the timing of a stream.

        :param sample_rate: Sample rate of the stream the source yields.
        """
        self.sample_rate = sample_rate
        self.epoch = 0
        self._spans: deque[_Span] = deque()
        self._end = 0
        self._source_end = 0
        self._epoch_changed = asyncio.Event()

    def add_audio(self, frames: int, play_ns: int) -> None:
        """
        Record audio the source yields next.

        :param frames: Number of frames yielded.
        :param play_ns: CLOCK_MONOTONIC_RAW ns at which the first frame should be played.
        """
        self._spans.append(_Span(self._end, frames, self._source_end, play_ns, self.epoch))
        self._end += frames
        self._source_end += frames
        self._trim_history()

    def add_silence(self, position: int, frames: int) -> None:
        """
        Record silence inserted into the stream.

        :param position: Frames of the source's own audio that precede the silence.
        :param frames: Number of frames of silence.
        """
        # the inserting side may still hold audio the source already yielded
        index = len(self._spans)
        while (
            index
            and self._spans[index - 1].play_ns is not None
            and self._spans[index - 1].source_start >= position
        ):
            index -= 1
        for i in range(index, len(self._spans)):
            self._spans[i].start += frames
        self._end += frames
        if index and (prev := self._spans[index - 1]).play_ns is None and prev.epoch == self.epoch:
            prev.frames += frames
        else:
            start = (
                self._spans[index].start - frames
                if index < len(self._spans)
                else self._end - frames
            )
            self._spans.insert(index, _Span(start, frames, position, None, self.epoch))
        self._trim_history()

    def new_epoch(self) -> None:
        """Mark all audio added so far as stale."""
        self.epoch += 1
        self._epoch_changed.set()
        self._epoch_changed = asyncio.Event()

    async def wait_for_new_epoch(self, epoch: int) -> int:
        """
        Wait until the epoch moves past the given one and return the new epoch.

        :param epoch: The epoch the caller last saw.
        """
        while self.epoch == epoch:
            await self._epoch_changed.wait()
        return self.epoch

    def play_time(self, start: int, frames: int, sample_rate: int) -> tuple[int, int] | None:
        """
        Return the first frame of a range the source timed since the last flush, and its time.

        The result is the frame's offset into the range and its CLOCK_MONOTONIC_RAW ns, or
        None when the range holds no such frame (inserted silence, stale or unknown audio).

        :param start: First frame of the range, counted at ``sample_rate``.
        :param frames: Number of frames in the range, counted at ``sample_rate``.
        :param sample_rate: Sample rate the consumer counts in, e.g. after resampling.
        """
        ratio = self.sample_rate / sample_rate
        begin, stop = start * ratio, (start + frames) * ratio
        for span in self._spans:
            if span.end <= begin:
                continue
            if span.start >= stop:
                break
            if span.play_ns is None or span.epoch != self.epoch:
                continue
            position = max(begin, span.start)
            offset = round(position / ratio) - start
            if offset >= frames:
                break
            return offset, span.play_ns + round((position - span.start) * 1e9 / self.sample_rate)
        return None

    def prune(self, position: int, sample_rate: int) -> None:
        """
        Forget the play times before a position the consumer has passed.

        :param position: Frame position, counted at ``sample_rate``.
        :param sample_rate: Sample rate the consumer counts in.
        """
        position_frames = position * self.sample_rate / sample_rate
        while self._spans and self._spans[0].end <= position_frames:
            self._spans.popleft()

    def _trim_history(self) -> None:
        keep_from = self._end - _MAX_HISTORY_SECONDS * self.sample_rate
        while self._spans and self._spans[0].end <= keep_from:
            self._spans.popleft()
