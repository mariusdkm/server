"""Tests for the shairport-sync metadata pipe reader."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import pathlib
import time
from collections.abc import Callable
from typing import Any

import pytest

from music_assistant.providers.airplay_receiver.mdns import ServiceRegistration
from music_assistant.providers.airplay_receiver.metadata import MetadataReader


@pytest.fixture
def pipe_path(tmp_path: pathlib.Path) -> str:
    """Create a metadata FIFO like the provider does."""
    path = str(tmp_path / "metadata_pipe")
    os.mkfifo(path)
    return path


async def _write_marker(pipe_path: str, marker: str, timeout: float = 5.0) -> None:
    """Write a marker using a short-lived FIFO writer, like a sessioncontrol hook does."""
    # O_NONBLOCK so a regression leaving the FIFO readerless fails the test instead of
    # hanging it: opening for write raises ENXIO while no reader is attached.
    async with asyncio.timeout(timeout):
        while True:
            try:
                fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                await asyncio.sleep(0.05)
                continue
            try:
                os.write(fd, f"{marker}\n".encode())
            finally:
                os.close(fd)
            return


async def _wait_for(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


async def test_pipe_is_open_when_start_returns(pipe_path: str) -> None:
    """start() attaches to the FIFO so hook writers never block on a missing reader."""
    reader = MetadataReader(pipe_path, logging.getLogger("test"), None)
    await reader.start()
    try:
        # Opening for write fails with ENXIO while no reader is attached.
        fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
    finally:
        await reader.stop()


async def test_hook_marker_delivered(pipe_path: str) -> None:
    """A sessioncontrol hook marker is delivered as a play_state update."""
    updates: list[dict[str, Any]] = []
    reader = MetadataReader(pipe_path, logging.getLogger("test"), updates.append)
    await reader.start()
    try:
        await _write_marker(pipe_path, "MA_PLAY_BEGIN")
        await _wait_for(lambda: bool(updates))
        assert updates == [{"play_state": "playing"}]
    finally:
        await reader.stop()


async def test_markers_after_writer_close(pipe_path: str) -> None:
    """Accept markers from hook writers that connect after an earlier writer closed."""
    updates: list[dict[str, Any]] = []
    reader = MetadataReader(pipe_path, logging.getLogger("test"), updates.append)
    await reader.start()
    try:
        await _write_marker(pipe_path, "MA_PLAY_BEGIN")
        await _wait_for(lambda: len(updates) == 1)
        await _write_marker(pipe_path, "MA_PLAY_END")
        await _wait_for(lambda: len(updates) == 2)
        assert updates == [{"play_state": "playing"}, {"play_state": "stopped"}]
    finally:
        await reader.stop()


async def test_reader_stays_idle_after_writer_close(pipe_path: str) -> None:
    """A FIFO with no writers left must not keep the event loop busy."""
    updates: list[dict[str, Any]] = []
    reader = MetadataReader(pipe_path, logging.getLogger("test"), updates.append)
    await reader.start()
    try:
        await _write_marker(pipe_path, "MA_PLAY_BEGIN")
        # The marker arriving proves the reader consumed it and the hook writer is gone.
        await _wait_for(lambda: bool(updates))
        cpu_before = time.process_time()
        await asyncio.sleep(0.5)
        # Only epoll reports a writerless FIFO readable forever, so this guard is
        # meaningful on Linux and passes trivially on macOS/kqueue.
        assert time.process_time() - cpu_before < 0.1
    finally:
        await reader.stop()


async def test_reader_keeps_pipe_attached_across_restart(
    pipe_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected read error restarts the loop without ever detaching from the FIFO."""
    updates: list[dict[str, Any]] = []
    reader = MetadataReader(pipe_path, logging.getLogger("test"), updates.append)
    real_read = os.read
    fail_once = True

    def flaky_read(fd: int, size: int) -> bytes:
        nonlocal fail_once
        if fd == reader._fd and fail_once:
            fail_once = False
            raise ValueError("unexpected error")
        return real_read(fd, size)

    monkeypatch.setattr(os, "read", flaky_read)
    await reader.start()
    try:
        await _write_marker(pipe_path, "MA_PLAY_BEGIN")
        await _wait_for(lambda: not fail_once)
        # Well inside the restart backoff: a hook writer must still find a reader here.
        await asyncio.sleep(0.2)
        os.close(os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK))
        await _write_marker(pipe_path, "MA_PLAY_END")
        await _wait_for(lambda: updates[-1:] == [{"play_state": "stopped"}], timeout=5.0)
    finally:
        await reader.stop()


def _xml_item(item_type: str, code: str, data: str) -> str:
    """Build a metadata item the way shairport-sync writes it to the pipe."""
    payload = base64.b64encode(data.encode()).decode()
    return (
        f"<item><type>{item_type.encode().hex()}</type><code>{code.encode().hex()}</code>"
        f'<length>{len(data)}</length>\n<data encoding="base64">\n{payload}</data></item>\n'
    )


async def test_service_registration_delivered(pipe_path: str) -> None:
    """A service from the "metadata" mDNS backend is delivered for publishing."""
    updates: list[dict[str, Any]] = []
    reader = MetadataReader(pipe_path, logging.getLogger("test"), updates.append)
    await reader.start()
    try:
        await _write_marker(
            pipe_path,
            _xml_item("ssnc", "svrg", "service=_airplay._tcp\nname=Kitchen\nport=7123\ntxt=a=1\n"),
        )
        await _wait_for(lambda: bool(updates))
        assert updates == [
            {"mdns_service": ServiceRegistration("_airplay._tcp", "Kitchen", 7123, {"a": "1"})}
        ]
    finally:
        await reader.stop()


async def test_malformed_service_registration_ignored() -> None:
    """A service that cannot be advertised is not passed on."""
    updates: list[dict[str, Any]] = []
    reader = MetadataReader("/nonexistent", logging.getLogger("test"), updates.append)
    await reader._process_metadata_item("ssnc", "svup", "name=Kitchen\n")
    assert updates == []


async def test_airplay2_resume_clears_pause() -> None:
    """AirPlay 2 resumes with 'pres' where AirPlay 1 sends 'prsm'."""
    reader = MetadataReader("/nonexistent", logging.getLogger("test"), None)
    for code in ("prsm", "pres"):  # codespell:ignore pres
        await reader._process_metadata_item("ssnc", "paus", "")
        assert reader._current_metadata["paused"] is True
        await reader._process_metadata_item("ssnc", code, "")
        assert reader._current_metadata["paused"] is False
