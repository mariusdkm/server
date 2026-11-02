"""Unit tests for the shared PTP clock daemon of the AirPlay providers."""

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from music_assistant.constants import VERBOSE_LOG_LEVEL
from music_assistant.providers.airplay.constants import (
    PTP_DAEMON_WARN_BURST,
    PTP_DAEMON_WARN_WINDOW,
)
from music_assistant.providers.airplay.helpers import CLI_CLOCK_OFFSET_NS
from music_assistant.providers.airplay.ptp_daemon import PtpDaemon

if TYPE_CHECKING:
    import pytest

MODULE = "music_assistant.providers.airplay.ptp_daemon"

# A representative readiness line emitted by `cliairplay --ptp-daemon` on the
# real device once it has bound 319/320 and opened its control channel.
DAEMON_UP_LINE = (
    "[15:44:56.101] ap2_ptp_run_daemon:1467 [PTP] daemon up: engine on 319/320, "
    "clock in /cliairplay-ptp, control on 127.0.0.1:9010"
)
TRACE_LINE = "[15:44:56.101] [PTP] slave offset seq=7 error=0.000012"


def _mass(source_ip: str | None = None) -> MagicMock:
    mass = MagicMock()
    mass.server_id = "aabbccdd11223344eeff"
    mass.streams.get_source_ip = AsyncMock(return_value=source_ip)

    def _consume_task(coro: object) -> MagicMock:
        if asyncio.iscoroutine(coro):
            coro.close()
        return MagicMock()

    mass.create_task.side_effect = _consume_task
    return mass


def _daemon(source_ip: str | None = None) -> PtpDaemon:
    daemon = PtpDaemon(_mass(source_ip))
    daemon.logger = logging.getLogger("test.airplay.ptp")
    return daemon


def _live_process() -> MagicMock:
    """Build a process mock that remains alive until its wait task is cancelled."""
    process = MagicMock()
    process.closed = False
    process.returncode = None
    process.close = AsyncMock()

    async def _wait_forever() -> int:
        await asyncio.Event().wait()
        return 0

    process.wait = AsyncMock(side_effect=_wait_forever)
    return process


@contextmanager
def _spawn() -> Iterator[MagicMock]:
    """Patch the binary lookup and process spawn; yield the AsyncProcess class mock."""

    def _new_process(*_args: object, **_kwargs: object) -> MagicMock:
        process = _live_process()
        process.start = AsyncMock()
        return process

    with (
        patch(f"{MODULE}.get_cli_binary", AsyncMock(return_value="/bin/cliairplay")),
        patch(f"{MODULE}.AsyncProcess", side_effect=_new_process) as process_cls,
    ):
        yield process_cls


def _logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("test.airplay.owner")
    # The spawn mirrors the live logger level into --debug flags; pin the level
    # so suite ordering cannot leak extra args into an assertion.
    logger.setLevel(level)
    return logger


# --- spawning ------------------------------------------------------------------


async def test_spawn_advertises_server_identity() -> None:
    """The daemon is handed the server's DACP id as its clock identity."""
    daemon = _daemon()
    with _spawn() as process_cls:
        await daemon.acquire("airplay", _logger())

    assert process_cls.call_args.args[0] == [
        "/bin/cliairplay",
        "--ptp-daemon",
        "--dacp",
        "AABBCCDD11223344",
        "--clock-offset-ns",
        str(CLI_CLOCK_OFFSET_NS),
    ]


async def test_spawn_pins_operator_configured_interface() -> None:
    """One daemon serves every player, so only an explicit operator pin narrows it."""
    daemon = _daemon("192.168.1.5")
    with _spawn() as process_cls:
        await daemon.acquire("airplay", _logger())

    args = process_cls.call_args.args[0]
    assert args[args.index("--if") + 1] == "192.168.1.5"


async def test_spawn_quiet_unless_verbose_and_opted_in() -> None:
    """The per-packet trace needs verbose logging AND the opt-in toggle."""
    for level, trace, traced in (
        (logging.DEBUG, True, False),
        (VERBOSE_LOG_LEVEL, False, False),
        (VERBOSE_LOG_LEVEL, True, True),
    ):
        daemon = _daemon()
        with _spawn() as process_cls:
            await daemon.acquire("airplay", _logger(level), trace=trace)
        args = process_cls.call_args.args[0]
        assert (args[-2:] == ["--debug", "10"]) is traced


# --- holders -------------------------------------------------------------------


async def test_one_daemon_per_server() -> None:
    """Both providers of a server get the same daemon."""
    mass = _mass()
    assert PtpDaemon.get(mass) is PtpDaemon.get(mass)
    assert PtpDaemon.get(mass) is not PtpDaemon.get(_mass())


async def test_runs_while_anyone_holds_it() -> None:
    """The second holder reuses the daemon; the last release stops it."""
    daemon = _daemon()
    with _spawn() as process_cls:
        await daemon.acquire("airplay", _logger())
        await daemon.acquire("airplay_receiver", _logger())
    assert process_cls.call_count == 1
    process = cast("MagicMock", daemon._process)

    await daemon.release("airplay")
    assert daemon.running
    await daemon.release("unknown")
    assert cast("bool", daemon.running)
    await daemon.release("airplay_receiver")
    assert not cast("bool", daemon.running)
    process.close.assert_awaited_once()


# --- readiness -----------------------------------------------------------------


def test_ready_event_set_on_daemon_up_line() -> None:
    """The readiness event is set when the daemon prints its 'daemon up' line."""
    daemon = _daemon()
    daemon._ready = asyncio.Event()
    daemon._handle_line(DAEMON_UP_LINE)
    assert daemon._ready.is_set()


def test_line_without_marker_leaves_event_clear() -> None:
    """A normal daemon log line does not trip the readiness event."""
    daemon = _daemon()
    daemon._ready = asyncio.Event()
    daemon._handle_line("[15:44:56.101] ap2_ptp_engine_start:1173 [PTP] Engine started")
    assert not daemon._ready.is_set()


def test_line_handler_tolerates_no_event() -> None:
    """The line handler is a no-op for readiness before any daemon has started."""
    _daemon()._handle_line(DAEMON_UP_LINE)


async def test_wait_ready_true_when_signalled() -> None:
    """wait_ready returns True once readiness has been signalled."""
    daemon = _daemon()
    daemon._process = _live_process()
    daemon._ready = asyncio.Event()
    daemon._ready.set()
    assert await daemon.wait_ready(timeout=0.1) is True


async def test_wait_ready_false_without_daemon() -> None:
    """With no daemon ever started, readiness is False."""
    assert await _daemon().wait_ready(timeout=0.1) is False


async def test_wait_ready_times_out_when_never_ready() -> None:
    """A spawned-but-not-ready daemon yields False after the bounded wait."""
    daemon = _daemon()
    daemon._process = _live_process()
    daemon._ready = asyncio.Event()
    assert await daemon.wait_ready(timeout=0.05) is False


async def test_wait_ready_false_fast_when_process_gone_or_exited() -> None:
    """A daemon that has exited short-circuits to False without burning the timeout."""
    for process in (None, MagicMock(closed=True, returncode=1)):
        daemon = _daemon()
        daemon._process = process
        daemon._ready = asyncio.Event()
        started = time.monotonic()
        assert await daemon.wait_ready(timeout=5.0) is False
        assert time.monotonic() - started < 1.0


async def test_wait_ready_rejects_stale_ready_event() -> None:
    """A stale readiness signal cannot make an exited daemon appear usable."""
    daemon = _daemon()
    daemon._process = MagicMock(closed=True, returncode=1)
    daemon._ready = asyncio.Event()
    daemon._ready.set()
    assert await daemon.wait_ready(timeout=0.1) is False


async def test_wait_ready_signalled_during_wait() -> None:
    """A readiness signal that arrives while waiting resolves the wait to True."""
    daemon = _daemon()
    daemon._process = _live_process()
    daemon._ready = asyncio.Event()

    async def _signal_soon() -> None:
        await asyncio.sleep(0.02)
        daemon._handle_line(DAEMON_UP_LINE)

    signaller = asyncio.create_task(_signal_soon())
    try:
        assert await daemon.wait_ready(timeout=1.0) is True
    finally:
        await signaller


async def test_stop_awaits_cancelled_stdout_reader() -> None:
    """Stopping waits for the cancelled stdout reader before process cleanup."""
    daemon = _daemon()
    reader_started = asyncio.Event()

    async def _stdout_reader() -> None:
        reader_started.set()
        await asyncio.Event().wait()

    reader_task = asyncio.create_task(_stdout_reader())
    daemon._stdout_task = reader_task
    await reader_started.wait()
    await daemon._stop()
    assert reader_task.cancelled()


# --- problem lines -------------------------------------------------------------


def test_problem_lines_are_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    """A repeating trace line matching a marker must not fill the log at WARNING."""
    daemon = _daemon()
    with caplog.at_level(logging.WARNING):
        for _ in range(PTP_DAEMON_WARN_BURST + 20):
            daemon._handle_line(TRACE_LINE)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == PTP_DAEMON_WARN_BURST


def test_reports_what_it_suppressed(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count of suppressed lines is reported once the window rolls over."""
    daemon = _daemon()
    clock = 1000.0
    monkeypatch.setattr(f"{MODULE}.time.monotonic", lambda: clock)
    with caplog.at_level(logging.WARNING):
        for _ in range(PTP_DAEMON_WARN_BURST + 3):
            daemon._handle_line(TRACE_LINE)
        caplog.clear()
        clock += PTP_DAEMON_WARN_WINDOW + 1
        daemon._handle_line(TRACE_LINE)

    messages = [record.getMessage() for record in caplog.records]
    assert any("3 further problem line(s) were suppressed" in message for message in messages)
    assert any(message.endswith(TRACE_LINE) for message in messages)


def test_warn_window_runs_from_its_first_line(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """time.monotonic() counts from boot, so a server started early must still get a full window."""
    daemon = _daemon()
    clock = 5.0  # five seconds of uptime
    monkeypatch.setattr(f"{MODULE}.time.monotonic", lambda: clock)
    for _ in range(PTP_DAEMON_WARN_BURST + 2):
        daemon._handle_line(TRACE_LINE)

    clock += PTP_DAEMON_WARN_WINDOW - 4
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        daemon._handle_line(TRACE_LINE)
    assert caplog.text == ""


def test_restart_gets_a_fresh_warning_budget(caplog: pytest.LogCaptureFixture) -> None:
    """A replacement daemon's startup failure must not be eaten by the old one's budget."""
    daemon = _daemon()
    for _ in range(PTP_DAEMON_WARN_BURST + 2):
        daemon._handle_line(TRACE_LINE)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        daemon._reset_warn_budget()
        daemon._handle_line("[15:44:57.002] [PTP] Cannot bind UDP 319: Permission denied")

    messages = [record.getMessage() for record in caplog.records]
    assert any("2 further problem line(s) from the previous daemon" in m for m in messages)
    assert any("Cannot bind UDP 319" in m for m in messages)


# --- exits ---------------------------------------------------------------------


def _exited(daemon: PtpDaemon, returncode: int) -> tuple[MagicMock, MagicMock]:
    """Make the daemon's process one that exited; return it and the daemon's logger."""
    process = MagicMock()
    process.wait = AsyncMock(return_value=returncode)
    ready = asyncio.Event()
    ready.set()
    logger = MagicMock()
    daemon.logger = logger
    daemon._process = process
    daemon._ready = ready
    daemon._started = time.monotonic()
    return process, logger


async def test_bind_failure_degrades_without_restart() -> None:
    """A privileged-port bind failure leaves streams on NTP timing, without a restart."""
    daemon = _daemon()
    process, logger = _exited(daemon, 2)
    with patch.object(daemon, "_start", new_callable=AsyncMock) as restart:
        await daemon._monitor(process)

    restart.assert_not_awaited()
    assert not await daemon.wait_ready(timeout=0)
    warning = logger.warning
    warning.assert_called_once()
    assert warning.call_args.args[1] == 2
    assert "CAP_NET_BIND_SERVICE" in warning.call_args.args[0]


async def test_crash_restarts_once() -> None:
    """A crash after startup restarts the daemon once."""
    daemon = _daemon()
    process, _ = _exited(daemon, 1)
    daemon._started = time.monotonic() - 60
    with patch.object(daemon, "_start", new_callable=AsyncMock) as restart:
        await daemon._monitor(process)
        restart.assert_awaited_once_with()
        process, _ = _exited(daemon, 1)
        daemon._started = time.monotonic() - 60
        await daemon._monitor(process)
    restart.assert_awaited_once()
