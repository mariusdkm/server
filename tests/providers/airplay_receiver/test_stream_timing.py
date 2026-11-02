"""Tests for the play times the AirPlay receiver keeps per audio stream."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from weakref import WeakKeyDictionary

from music_assistant_models.media_items.audio_format import AudioFormat

from music_assistant.providers.airplay_receiver import AirPlayReceiverProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant.helpers.live_timing import LiveTiming

_SECOND_NS = 1_000_000_000


async def _received_audio(timing: LiveTiming) -> AsyncGenerator[bytes]:
    timing.add_audio(100, 10 * _SECOND_NS)
    yield b"\x00" * 400


def _provider() -> AirPlayReceiverProvider:
    """Return a provider with one running receiver, without starting any daemon."""
    provider = object.__new__(AirPlayReceiverProvider)
    provider._decoded_audio_format = AudioFormat(sample_rate=1000, bit_depth=16, channels=2)
    provider._stream_timings = WeakKeyDictionary()
    daemon = MagicMock()
    daemon.audio_stream = None
    daemon.audio_reader.timed = True
    daemon.audio_reader.iter_audio = _received_audio
    provider._daemons = {"receiver": daemon}
    return provider


async def test_silence_of_a_superseded_stream_stays_with_it() -> None:
    """Silence reported for an old stream does not shift the new stream's play times."""
    provider = _provider()
    streamdetails = MagicMock(item_id="receiver")
    old = provider.get_audio_stream(streamdetails)
    await anext(old)
    old_timing = provider.get_audio_source_timing(streamdetails)
    # the source restarts its stream with the same stream details
    new = provider.get_audio_stream(streamdetails)
    await anext(new)
    new_timing = provider.get_audio_source_timing(streamdetails)
    assert new_timing is not None
    assert new_timing is not old_timing

    # the old stream's keepalive is still winding down, at its own positions
    provider.on_audio_source_silence(streamdetails, old, 0, 50)

    assert new_timing.play_time(0, 100, 1000) == (0, 10 * _SECOND_NS)
    provider.on_audio_source_silence(streamdetails, new, 100, 50)
    assert new_timing.play_time(100, 50, 1000) is None
    await old.aclose()
    await new.aclose()
