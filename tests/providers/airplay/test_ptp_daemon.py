"""Unit tests for the shared PTP clock daemon of the AirPlay providers."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from music_assistant.constants import VERBOSE_LOG_LEVEL
from music_assistant.helpers.timeline_clock import TimelineClock
from music_assistant.providers.airplay.constants import (
    PTP_DAEMON_WARN_BURST,
    PTP_DAEMON_WARN_WINDOW,
)
from music_assistant.providers.airplay.ptp_daemon import PTP_DAEMON_EXIT_NQPTP, PtpDaemon

if TYPE_CHECKING:
    import pytest

MODULE = "music_assistant.providers.airplay.ptp_daemon"

# A representative readiness line emitted by `cliairplay --ptp-daemon` on the
# real device (from a live DEBUG log) once it has bound 319/320 and opened its
# control channel.
DAEMON_UP_LINE = (
    "[15:44:56.101] ap2_ptp_run_daemon:1467 [PTP] daemon up: engine on 319/320, "
    "clock in /cliairplay-ptp, control on 127.0.0.1:9010"
)
DAEMON_UP_NQPTP_LINE = DAEMON_UP_LINE + ", nqptp=on"
TRACE_LINE = "[15:44:56.101] [PTP] slave offset seq=7 error=0.000012"


def _mass(source_ip: str | None = None) -> MagicMock:
    mass = MagicMock()
    mass.server_id = "aabbccdd11223344eeff"
    mass.streams.get_source_ip = AsyncMock(return_value=source_ip)
    mass.timeline_clock = TimelineClock()

    def _consume_task(coro: object) -> MagicMock:
        if asyncio.iscoroutine(coro):
            coro.close()
        return MagicMock()

    mass.create_task.side_effect = _consume_task
    return mass


def _daemon(source_ip: str | None = None) -> PtpDaemon:
    daemon = PtpDaemon(_mass(source_ip))
    daemon.logger = logging.getLogger("test.airplay.ptp")
    # The spawn mirrors the live logger level into --debug flags; pin the
    # level so suite ordering (other tests raising verbosity) cannot leak
    # extra args into the assertions.
    daemon.logger.setLevel(logging.INFO)
    return daemon


def _live_process() -> MagicMock:
    """Build a process mock that remains alive until its wait task is cancelled."""
    process = MagicMock()
    process.closed = False
    process.returncode = None
    process.start = AsyncMock()
    process.close = AsyncMock()
    process.read = AsyncMock(return_value=b"")

    async def _no_lines() -> AsyncIterator[str]:
        return
        yield

    process.iter_stderr = _no_lines

    async def _wait_forever() -> int:
        await asyncio.Event().wait()
        return 0

    process.wait = AsyncMock(side_effect=_wait_forever)
    return process


@contextmanager
def _spawn(*processes: MagicMock, help_text: bytes = b"--nqptp-compat") -> Iterator[MagicMock]:
    """Patch the binary lookup and process spawn; yield the AsyncProcess class mock."""
    side_effect: Any = list(processes) if processes else lambda *_a, **_k: _live_process()
    with (
        patch(f"{MODULE}.get_cli_binary", AsyncMock(return_value="/bin/cliairplay")),
        patch(f"{MODULE}.check_output", AsyncMock(return_value=(0, help_text))),
        patch(f"{MODULE}.AsyncProcess", side_effect=side_effect) as process_cls,
    ):
        yield process_cls


def _mass_of(daemon: PtpDaemon) -> MagicMock:
    """Return the mocked server of a daemon built by :func:`_daemon`."""
    return cast("MagicMock", daemon.mass)


def _process_of(daemon: PtpDaemon) -> MagicMock:
    """Return the mocked process the daemon runs."""
    assert daemon._process is not None
    return cast("MagicMock", daemon._process)


def _eager_tasks(daemon: PtpDaemon) -> list[asyncio.Task[None]]:
    """Run the daemon's tasks eagerly, as the server does, and collect them."""
    tasks: list[asyncio.Task[None]] = []

    def _create_task(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        tasks.append(task := asyncio.Task(coro, eager_start=True))
        return task

    _mass_of(daemon).create_task.side_effect = _create_task
    return tasks


# --- spawning ----------------------------------------------------------------------


async def test_spawn_advertises_the_server_identity_on_the_timeline() -> None:
    """The daemon gets the server's DACP id as its clock identity and runs on the timeline."""
    daemon = _daemon()
    clock: TimelineClock = daemon.mass.timeline_clock
    clock.set_rate(857_000)

    with _spawn() as process_cls:
        await daemon._start()

    assert process_cls.call_args.args[0] == [
        "/bin/cliairplay",
        "--ptp-daemon",
        "--dacp",
        "AABBCCDD11223344",
        "--clock-offset-ns",
        str(clock.offset_ns + clock.unix_offset_ns),
        "--clock-rate-ppb",
        "857000",
    ]


async def test_spawn_pins_operator_configured_interface() -> None:
    """One daemon serves every player, so only an explicit operator pin narrows it."""
    daemon = _daemon(source_ip="192.168.1.5")

    with _spawn() as process_cls:
        await daemon._start()

    args = process_cls.call_args.args[0]
    assert args[args.index("--if") + 1] == "192.168.1.5"


async def test_spawn_quiet_at_debug_level() -> None:
    """A normal debug session keeps the daemon quiet (no per-packet tracing)."""
    daemon = _daemon()
    daemon.logger.setLevel(logging.DEBUG)
    daemon._trace = True

    with _spawn() as process_cls:
        await daemon._start()

    assert "--debug" not in process_cls.call_args.args[0]


async def test_spawn_traces_at_verbose_level_with_the_opt_in() -> None:
    """The daemon's per-packet trace needs verbose logging AND a holder's opt-in."""
    daemon = _daemon()
    daemon.logger.setLevel(VERBOSE_LOG_LEVEL)
    daemon._trace = True

    with _spawn() as process_cls:
        await daemon._start()
    assert process_cls.call_args.args[0][-2:] == ["--debug", "10"]

    # Without the opt-in the daemon stays quiet even on a verbose session
    # (its ~10 lines/s timing trace would otherwise flood every verbose log).
    daemon._trace = False
    with _spawn() as process_cls:
        await daemon._start()
    assert "--debug" not in process_cls.call_args.args[0]


# --- holders -----------------------------------------------------------------------


async def test_first_holder_starts_the_daemon_and_the_last_stops_it() -> None:
    """The daemon runs while anyone holds it, and is spawned only once."""
    daemon = _daemon()
    logger = logging.getLogger("test.airplay.holder")

    with _spawn() as process_cls:
        await daemon.acquire("airplay", logger)
        await daemon.acquire("airplay_receiver", logger)
    assert process_cls.call_count == 1
    process = _process_of(daemon)

    await daemon.release("airplay")
    running_after_first = daemon.running
    process.close.assert_not_awaited()

    await daemon.release("airplay_receiver")
    assert running_after_first
    assert not daemon.running
    process.close.assert_awaited_once()


async def test_release_of_an_unknown_holder_keeps_the_daemon() -> None:
    """Only a holder can release the daemon, and only once."""
    daemon = _daemon()

    with _spawn():
        await daemon.acquire("airplay", daemon.logger)
    await daemon.release("someone else")
    await daemon.release("airplay")
    await daemon.release("airplay")

    assert not daemon.running


async def test_one_daemon_per_server() -> None:
    """Every provider of a server gets the same daemon."""
    mass, other = _mass(), _mass()

    assert PtpDaemon.get(mass) is PtpDaemon.get(mass)
    assert PtpDaemon.get(mass) is not PtpDaemon.get(other)


async def test_last_release_unregisters_from_the_timeline_clock() -> None:
    """A daemon nobody holds no longer acts on a new rate."""
    daemon = _daemon()
    clock: TimelineClock = daemon.mass.timeline_clock

    with _spawn():
        await daemon.acquire("airplay", daemon.logger)
    await daemon.release("airplay")
    _mass_of(daemon).create_task.reset_mock()
    clock.set_rate(857_000)

    _mass_of(daemon).create_task.assert_not_called()


# --- nqptp compatibility ------------------------------------------------------------


async def test_nqptp_holder_restarts_a_daemon_without_it() -> None:
    """A holder that needs nqptp compatibility gets it, once."""
    daemon = _daemon()
    with _spawn() as process_cls:
        await daemon.acquire("airplay", daemon.logger)
        first = _process_of(daemon)
        await daemon.acquire("airplay_receiver", daemon.logger, nqptp_compat=True)
        await daemon.acquire("other", daemon.logger)

    assert process_cls.call_count == 2
    assert "--nqptp-compat" not in process_cls.call_args_list[0].args[0]
    assert "--nqptp-compat" in process_cls.call_args_list[1].args[0]
    first.close.assert_awaited_once()

    # Releasing the nqptp holder does not restart the daemon again.
    with _spawn() as process_cls:
        await daemon.release("airplay_receiver")
    process_cls.assert_not_called()
    assert daemon.running


async def test_nqptp_holder_first_starts_with_it() -> None:
    """A daemon started for an nqptp holder starts with compatibility."""
    daemon = _daemon()
    with _spawn() as process_cls:
        await daemon.acquire("airplay_receiver", daemon.logger, nqptp_compat=True)
        await daemon.acquire("airplay", daemon.logger)
    assert process_cls.call_count == 1
    assert "--nqptp-compat" in process_cls.call_args.args[0]


async def test_old_binary_without_nqptp_compat() -> None:
    """A cliairplay without --nqptp-compat runs the daemon without it."""
    daemon = _daemon()
    with _spawn(help_text=b"--ptp-daemon") as process_cls:
        await daemon.acquire("airplay_receiver", daemon.logger, nqptp_compat=True)
    assert "--nqptp-compat" not in process_cls.call_args.args[0]


async def test_new_timeline_rate_keeps_nqptp_compatibility() -> None:
    """A daemon restarted on a new rate keeps standing in for nqptp."""
    daemon = _daemon()
    clock: TimelineClock = daemon.mass.timeline_clock
    with _spawn():
        await daemon.acquire("airplay_receiver", daemon.logger, nqptp_compat=True)
    _eager_tasks(daemon)

    with _spawn() as process_cls:
        clock.set_rate(857_000)
        assert daemon._restart_task is not None
        await daemon._restart_task

    assert "--nqptp-compat" in process_cls.call_args.args[0]


async def test_nqptp_active_only_when_the_ready_line_says_so() -> None:
    """Readiness reports whether the daemon stands in for nqptp."""
    daemon = _daemon()
    daemon._process = _live_process()
    daemon._ready = asyncio.Event()
    daemon._handle_line(DAEMON_UP_LINE)
    assert daemon.ready
    assert not daemon.nqptp_active

    daemon._ready.clear()
    daemon._handle_line(DAEMON_UP_NQPTP_LINE)
    assert daemon.nqptp_active


# --- readiness and output ------------------------------------------------------------


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

    daemon._handle_line(
        "[15:44:56.101] ap2_ptp_engine_start:1173 [PTP] Engine started on UDP 319/320"
    )

    assert not daemon._ready.is_set()


def test_line_handler_tolerates_no_event() -> None:
    """The line handler is a no-op for readiness before any daemon has started."""
    # Must not raise even when the readiness gate does not exist yet.
    _daemon()._handle_line(DAEMON_UP_LINE)


def test_problem_lines_are_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    """A repeating trace line matching a marker must not fill the log at WARNING."""
    daemon = _daemon()

    with caplog.at_level(logging.WARNING):
        for _ in range(PTP_DAEMON_WARN_BURST + 20):
            daemon._handle_line(TRACE_LINE)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == PTP_DAEMON_WARN_BURST


def test_daemon_reports_what_it_suppressed(
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
    # and the window reopens for real problems
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

    # 56s into the window, so it must not have rolled over yet
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


async def test_taken_nqptp_port_keeps_ptp_without_compat() -> None:
    """With a real nqptp on the host, the daemon restarts for the sender only."""
    daemon = _daemon()
    daemon._holders = {"airplay_receiver": True}
    daemon._compat = True
    process = MagicMock()
    process.wait = AsyncMock(return_value=PTP_DAEMON_EXIT_NQPTP)
    daemon._process = process
    daemon._started = time.monotonic()
    daemon.logger = logger = MagicMock()

    with patch.object(daemon, "_start", new_callable=AsyncMock) as restart:
        await daemon._monitor(process)

    restart.assert_awaited_once_with(False)
    assert not daemon._wants_compat()
    assert "9000" in logger.warning.call_args.args[0]


async def test_crash_restarts_once_with_the_same_mode() -> None:
    """A crash after startup restarts the daemon once, keeping nqptp compatibility."""
    daemon = _daemon()
    daemon._compat = True
    daemon.logger = MagicMock()

    def _crashed() -> MagicMock:
        process = MagicMock()
        process.wait = AsyncMock(return_value=1)
        daemon._process = process
        daemon._started = time.monotonic() - 60
        return process

    with patch.object(daemon, "_start", new_callable=AsyncMock) as restart:
        await daemon._monitor(_crashed())
        restart.assert_awaited_once_with(True)
        await daemon._monitor(_crashed())
    restart.assert_awaited_once()


async def test_bind_failure_degrades_without_restart() -> None:
    """A privileged-port bind failure leaves streams on NTP timing, without a restart."""
    daemon = _daemon()
    process = MagicMock()
    process.wait = AsyncMock(return_value=2)
    daemon._ready = asyncio.Event()
    daemon._ready.set()
    daemon.logger = MagicMock()
    daemon._process = process
    daemon._started = time.monotonic()

    with patch.object(daemon, "_start", new_callable=AsyncMock) as restart:
        await daemon._monitor(process)

    restart.assert_not_awaited()
    assert not await daemon.wait_ready(timeout=0)
    warning = daemon.logger.warning
    warning.assert_called_once()
    assert warning.call_args.args[1] == 2
    assert "CAP_NET_BIND_SERVICE" in warning.call_args.args[0]


# --- the rate of the timeline -----------------------------------------------------------


async def test_new_timeline_rate_restarts_the_running_daemon() -> None:
    """A new rate of the timeline brings up a fresh daemon, which takes it on."""
    daemon = _daemon()
    clock: TimelineClock = daemon.mass.timeline_clock
    with _spawn():
        await daemon.acquire("airplay", daemon.logger)
    old = _process_of(daemon)
    assert daemon._ready is not None
    daemon._ready.set()
    _eager_tasks(daemon)

    with _spawn() as process_cls:
        clock.set_rate(857_000)
        # no stream may attach to the daemon on its way out
        assert not daemon._ready.is_set()
        assert daemon._restart_task is not None
        await daemon._restart_task

    old.close.assert_awaited_once()
    args = process_cls.call_args.args[0]
    assert args[args.index("--clock-rate-ppb") + 1] == "857000"
    assert daemon._process is not old


async def test_new_timeline_rate_leaves_a_stopped_daemon_alone() -> None:
    """A daemon that is not running starts on the current timeline whenever it starts."""
    daemon = _daemon()

    with patch.object(daemon, "_start", new_callable=AsyncMock) as start:
        await daemon._restart()

    start.assert_not_awaited()


async def test_a_holder_running_a_stream_holds_the_rate_back() -> None:
    """The restart waits while a holder pins the timeline, as its streams cannot follow."""
    daemon = _daemon()
    clock: TimelineClock = daemon.mass.timeline_clock
    with _spawn():
        await daemon.acquire("airplay", daemon.logger)
    clock.add_rate_listener(pinned=lambda: True)

    assert clock.rate_pinned


async def test_restart_stops_at_release() -> None:
    """A restart that the last release overtakes leaves no daemon behind."""
    daemon = _daemon()
    daemon._process = process = _live_process()

    async def _close() -> None:
        daemon._stop_requested = True

    process.close = AsyncMock(side_effect=_close)

    with patch.object(daemon, "_start", new_callable=AsyncMock) as start:
        await daemon._restart()

    start.assert_not_awaited()


async def test_release_waits_for_a_running_restart() -> None:
    """The last release returns only once a restart in flight is done, so no daemon outlives it."""
    daemon = _daemon()
    clock: TimelineClock = daemon.mass.timeline_clock
    with _spawn():
        await daemon.acquire("airplay", daemon.logger)
    release_close = asyncio.Event()
    _process_of(daemon).close = AsyncMock(side_effect=release_close.wait)
    _eager_tasks(daemon)

    with patch.object(daemon, "_start", new_callable=AsyncMock) as start:
        # the restart runs up to closing the old daemon
        clock.set_rate(857_000)
        release = asyncio.create_task(daemon.release("airplay"))
        await asyncio.sleep(0)
        released_early = release.done()
        release_close.set()
        await release

    assert not released_early
    start.assert_not_awaited()
    assert not daemon.running


async def test_daemon_spawned_on_an_old_rate_starts_over() -> None:
    """A new rate while the daemon spawns, with no daemon yet to restart, starts it over."""
    daemon = _daemon()
    clock: TimelineClock = daemon.mass.timeline_clock
    first, second = _live_process(), _live_process()

    async def _keeper_applies_a_rate() -> None:
        clock.set_rate(857_000)

    first.start = AsyncMock(side_effect=_keeper_applies_a_rate)

    with _spawn(first, second) as process_cls:
        await daemon._start()

    first_args, second_args = (call.args[0] for call in process_cls.call_args_list)
    assert first_args[first_args.index("--clock-rate-ppb") + 1] == "0"
    assert second_args[second_args.index("--clock-rate-ppb") + 1] == "857000"
    first.close.assert_awaited_once()
    assert daemon._process is second


async def test_daemon_spawned_during_release_is_closed() -> None:
    """A daemon that the last release overtakes while it spawns is closed again."""
    daemon = _daemon()
    process = _live_process()

    async def _holder_releases() -> None:
        daemon._stop_requested = True

    process.start = AsyncMock(side_effect=_holder_releases)

    with _spawn(process):
        await daemon._start()

    process.close.assert_awaited_once()
    assert daemon._process is None
    _mass_of(daemon).create_task.assert_not_called()


# --- waiting for readiness ---------------------------------------------------------------


async def test_wait_ready_true_when_signalled() -> None:
    """wait_ready returns True once readiness has been signalled."""
    daemon = _daemon()
    daemon._process = _live_process()
    daemon._ready = asyncio.Event()
    daemon._ready.set()

    assert await daemon.wait_ready(timeout=0.1) is True


async def test_wait_ready_false_without_daemon() -> None:
    """With no daemon ever started (event is None), readiness is False."""
    assert await _daemon().wait_ready(timeout=0.1) is False


async def test_wait_ready_times_out_when_never_ready() -> None:
    """A spawned-but-not-ready daemon yields False after the bounded wait."""
    daemon = _daemon()
    daemon._process = _live_process()
    daemon._ready = asyncio.Event()  # but readiness never signalled

    assert await daemon.wait_ready(timeout=0.05) is False


async def test_wait_ready_false_fast_when_daemon_gone() -> None:
    """A daemon that has exited short-circuits to False without burning the timeout."""
    daemon = _daemon()
    daemon._ready = asyncio.Event()  # stale gate left un-set

    started = time.monotonic()
    assert await daemon.wait_ready(timeout=5.0) is False
    assert time.monotonic() - started < 1.0


async def test_wait_ready_false_fast_when_process_exited() -> None:
    """An exited process object cannot keep readiness waiting on a stale event."""
    daemon = _daemon()
    daemon._process = MagicMock(closed=True, returncode=1)
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
