"""Tests for the silence nudge that unblocks a stopped AirPlay Receiver stream."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.providers.airplay_receiver import AirPlayReceiverProvider


def _pcm(sample_rate: int, bit_depth: int) -> AudioFormat:
    content_type = ContentType.from_bit_depth(bit_depth)
    return AudioFormat(
        content_type=content_type,
        codec_type=content_type,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=2,
    )


def _make_daemon(reader_attaches: bool) -> MagicMock:
    """
    Build a mock receiver daemon whose audio pipe reports the given reader state.

    :param reader_attaches: Whether a consumer attaches to the audio pipe.
    """
    daemon = MagicMock()
    daemon.audio_pipe.wait_for_reader = AsyncMock(return_value=reader_attaches)
    daemon.audio_pipe.write = AsyncMock(return_value=True)
    return daemon


@pytest.mark.parametrize(
    ("audio_format", "size"),
    [(_pcm(44100, 16), 176400), (_pcm(48000, 32), 384000)],
)
async def test_silence_nudge_waits_for_the_returning_reader(
    audio_format: AudioFormat, size: int
) -> None:
    """The nudge is delivered once the stream's consumer has reattached to the pipe."""
    provider = MagicMock()
    provider.logger = MagicMock()
    provider._decoded_audio_format = audio_format
    daemon = _make_daemon(reader_attaches=True)

    await AirPlayReceiverProvider._write_silence_to_unblock_stream(provider, daemon)

    daemon.audio_pipe.wait_for_reader.assert_awaited_once()
    # one second of the piped format, enough for the consumer to emit a chunk
    assert len(daemon.audio_pipe.write.await_args.args[0]) == size


async def test_silence_nudge_skipped_when_no_reader_returns() -> None:
    """A pipe that nothing reads back is left alone instead of taking a dropped write."""
    provider = MagicMock()
    provider.logger = MagicMock()
    daemon = _make_daemon(reader_attaches=False)

    await AirPlayReceiverProvider._write_silence_to_unblock_stream(provider, daemon)

    daemon.audio_pipe.write.assert_not_awaited()
