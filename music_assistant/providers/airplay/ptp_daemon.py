"""The shared PTP clock daemon (`cliairplay --ptp-daemon`) of a server."""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from contextlib import suppress
from typing import TYPE_CHECKING, Final

from music_assistant.constants import VERBOSE_LOG_LEVEL
from music_assistant.helpers.process import AsyncProcess, check_output

from .constants import CLI_PROBLEM_MARKERS, PTP_DAEMON_WARN_BURST, PTP_DAEMON_WARN_WINDOW
from .helpers import CLI_CLOCK_OFFSET_NS, get_cli_binary

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

# Marker the daemon prints once it has bound the privileged PTP ports (UDP
# 319/320) and opened its control channel. Until this line is seen the daemon
# is spawned but not yet able to serve shared-clock streams, so a group start
# before it would race the daemon's readiness.
PTP_DAEMON_READY_MARKER: Final[str] = "[PTP] daemon up"
# Appended to the ready line when the daemon also stands in for nqptp.
PTP_DAEMON_NQPTP_MARKER: Final[str] = "nqptp=on"
# Bounded wait for the daemon to report readiness before a stream session
# decides its timing source. Once ready the check returns immediately; only a
# session that starts while the daemon is still coming up (or never binds) pays
# any of this, and only up to the moment readiness is signalled.
PTP_DAEMON_READY_TIMEOUT: Final[float] = 3.0
# Exit code of a daemon whose nqptp control port (UDP 9000) is taken.
PTP_DAEMON_EXIT_NQPTP: Final[int] = 3

_DAEMONS: weakref.WeakKeyDictionary[MusicAssistant, PtpDaemon] = weakref.WeakKeyDictionary()


class PtpDaemon:
    """
    The one `cliairplay --ptp-daemon` of a server, shared by its owners.

    Only one process per host can bind UDP 319/320, so the AirPlay sender and the
    AirPlay receiver share this daemon. Each acquires it and releases it on unload;
    it runs while anyone holds it.
    """

    def __init__(self, mass: MusicAssistant) -> None:
        """
        Initialize the (not yet running) daemon.

        :param mass: The Music Assistant instance.
        """
        self.mass = mass
        self.logger = logging.getLogger(__name__)
        self._owners: dict[str, bool] = {}
        self._trace = False
        self._lock = asyncio.Lock()
        self._process: AsyncProcess | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._started: float = 0.0
        self._restarted = False
        self._stop_requested = False
        self._compat = False
        self._compat_blocked = False
        self._nqptp_active = False
        self._supports_compat: bool | None = None
        # Set once the running daemon reports it has bound 319/320 and opened its
        # control channel; created/cleared per daemon start so a crash+restart
        # re-gates readiness. None until the daemon is first started.
        self._ready: asyncio.Event | None = None
        # Rate-limit state for daemon lines promoted to WARNING (see
        # PTP_DAEMON_WARN_BURST).
        self._warn_window_start: float | None = None
        self._warns_in_window = 0
        self._warns_suppressed = 0

    @classmethod
    def get(cls, mass: MusicAssistant) -> PtpDaemon:
        """
        Return the daemon of this server.

        :param mass: The Music Assistant instance.
        """
        if (daemon := _DAEMONS.get(mass)) is None:
            daemon = _DAEMONS[mass] = cls(mass)
        return daemon

    @property
    def running(self) -> bool:
        """
        Return if the daemon process is alive.

        This reflects process liveness only. A daemon that never bound its ports
        stays alive and reports True here while every group silently degrades to
        NTP; use :attr:`ready` for the real state.
        """
        return (
            self._process is not None
            and not self._process.closed
            and self._process.returncode is None
        )

    @property
    def ready(self) -> bool:
        """Return whether the daemon has bound UDP 319/320 and is still alive."""
        return self.running and self._ready is not None and self._ready.is_set()

    @property
    def nqptp_active(self) -> bool:
        """Return whether the ready daemon also serves shairport-sync (`--nqptp-compat`)."""
        return self.ready and self._nqptp_active

    async def acquire(
        self,
        owner: str,
        logger: logging.Logger,
        *,
        nqptp_compat: bool = False,
        trace: bool = False,
    ) -> None:
        """
        Hold the daemon, starting it (or restarting it with nqptp compatibility) if needed.

        :param owner: A unique name of the holder, released with :meth:`release`.
        :param logger: Logger for the daemon's output when this call starts it.
        :param nqptp_compat: The holder needs the daemon to stand in for nqptp.
        :param trace: Run the daemon with its per-packet PTP trace.
        """
        async with self._lock:
            self._owners[owner] = nqptp_compat
            self._trace = self._trace or trace
            want_compat = self._wants_compat() and await self._compat_supported()
            if self.running and (self._compat or not want_compat):
                return
            if self.running:
                self.logger.info("Restarting the PTP clock daemon to serve shairport-sync")
                await self._stop()
            self.logger = logger
            self._restarted = False
            await self._start(want_compat)

    async def release(self, owner: str) -> None:
        """
        Release the daemon; the last holder stops it.

        :param owner: The name given to :meth:`acquire`.
        """
        async with self._lock:
            if self._owners.pop(owner, None) is None or self._owners:
                return
            await self._stop()
            self._trace = False
            self._compat_blocked = False

    async def wait_ready(self, timeout: float = PTP_DAEMON_READY_TIMEOUT) -> bool:
        """
        Wait until the daemon is ready to serve streams.

        Readiness means the daemon has bound the privileged PTP ports (UDP
        319/320) and opened its control channel - not merely that the process is
        alive.

        :param timeout: Maximum seconds to wait for the readiness signal.
        :return: True if the daemon has signalled readiness, False otherwise
            (never started, failed to bind, or not ready within the timeout).
        """
        event = self._ready
        daemon = self._process
        if event is None or daemon is None or daemon.closed or daemon.returncode is not None:
            return False
        if event.is_set():
            return True

        ready_task = asyncio.create_task(event.wait())
        exit_task = asyncio.create_task(daemon.wait())
        try:
            done, _ = await asyncio.wait(
                (ready_task, exit_task),
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            return (
                ready_task in done
                and event.is_set()
                and self._process is daemon
                and not daemon.closed
                and daemon.returncode is None
            )
        finally:
            ready_task.cancel()
            exit_task.cancel()
            await asyncio.gather(ready_task, exit_task, return_exceptions=True)

    def _wants_compat(self) -> bool:
        """Return whether any holder needs nqptp compatibility (and it is not blocked)."""
        return any(self._owners.values()) and not self._compat_blocked

    async def _compat_supported(self) -> bool:
        """Return whether the cliairplay binary knows `--nqptp-compat`."""
        if self._supports_compat is None:
            try:
                _, output = await check_output(await get_cli_binary(), "--help", timeout=10)
            except RuntimeError, OSError, TimeoutError:
                return False
            self._supports_compat = b"--nqptp-compat" in output
            if not self._supports_compat:
                self.logger.warning("This cliairplay cannot stand in for nqptp (no --nqptp-compat)")
        return self._supports_compat

    async def _start(self, nqptp_compat: bool) -> None:
        """Spawn the daemon (cliairplay --ptp-daemon)."""
        try:
            cli_binary = await get_cli_binary()
        except RuntimeError as err:
            self.logger.warning(
                "cliairplay binary unavailable (%s) - "
                "PTP timing is degraded, AirPlay 2 streams fall back to NTP",
                err,
            )
            return
        # Advertise the same grandmaster identity as the per-stream sessions
        # (which derive their PTP clock id from --dacp), so receivers see one
        # consistent clock whether the daemon or an in-process engine serves it.
        args = [
            cli_binary,
            "--ptp-daemon",
            "--dacp",
            self.mass.server_id[:16].upper(),
            "--clock-offset-ns",
            str(CLI_CLOCK_OFFSET_NS),
        ]
        if nqptp_compat:
            args.append("--nqptp-compat")
        if_arg = await self.mass.streams.get_source_ip()
        if if_arg:
            args += ["--if", if_arg]
        # The daemon runs quiet by default: its per-packet PTP tracing
        # (Announce/Sync/Delay_Req, ~10 lines/s) needs BOTH verbose logging and
        # the dedicated opt-in, so ordinary verbose sessions are not flooded
        # with timing chatter that only matters for clock-sync debugging.
        if self._trace and self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            args += ["--debug", "10"]
        # The binding the daemon ends up with is the first thing needed when triaging
        # timing issues from a user's log, and it is invisible otherwise: --if is left
        # out entirely for a default bind ip.
        self.logger.debug(
            "Starting shared PTP clock daemon: if=%s nqptp=%s",
            if_arg or "<all interfaces>",
            "on" if nqptp_compat else "off",
        )
        daemon = AsyncProcess(args, stdout=True, stderr=True, name="cliairplay-ptp-daemon")
        # (Re)gate readiness for this daemon instance: not ready until a reader
        # sees the "daemon up" line (a restart clears any previous readiness).
        if self._ready is None:
            self._ready = asyncio.Event()
        else:
            self._ready.clear()
        self._nqptp_active = False
        self._compat = nqptp_compat
        self._stop_requested = False
        self._reset_warn_budget()
        await daemon.start()
        self._process = daemon
        self._started = time.monotonic()
        daemon.attach_stderr_reader(self.mass.create_task(self._stderr_reader(daemon)))
        self._stdout_task = self.mass.create_task(self._stdout_reader(daemon))
        self.mass.create_task(self._monitor(daemon))

    async def _stop(self) -> None:
        """Terminate the daemon."""
        self._stop_requested = True
        if self._ready is not None:
            self._ready.clear()
        stdout_task = self._stdout_task
        if stdout_task and not stdout_task.done():
            stdout_task.cancel()
            with suppress(asyncio.CancelledError):
                await stdout_task
        if self._process and not self._process.closed:
            await self._process.close()
        self._process = None

    async def _stderr_reader(self, daemon: AsyncProcess) -> None:
        """Forward daemon stderr output to the log."""
        async for line in daemon.iter_stderr():
            self._handle_line(line)

    async def _stdout_reader(self, daemon: AsyncProcess) -> None:
        """Drain (and log) the daemon stdout pipe."""
        buffer = b""
        while chunk := await daemon.read(1024):
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                if line := raw_line.decode("utf-8", errors="ignore").strip():
                    self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        """Log a daemon output line and detect its readiness signal."""
        # Routine daemon output is verbose-only so it never floods a user's log
        # (the per-packet timing trace only runs at verbose in the first place).
        lowered = line.lower()
        if any(marker in lowered for marker in CLI_PROBLEM_MARKERS):
            self._warn_line(line)
        else:
            self.logger.log(VERBOSE_LOG_LEVEL, "PTP daemon: %s", line)
        # The readiness marker is matched on either pipe: the daemon's diagnostic
        # output is not contractually stdout-vs-stderr, so both readers feed this
        # handler and setting the event is idempotent.
        if (
            (event := self._ready) is not None
            and not event.is_set()
            and PTP_DAEMON_READY_MARKER in line
        ):
            self._nqptp_active = PTP_DAEMON_NQPTP_MARKER in line
            self.logger.debug("Shared PTP clock daemon reported ready")
            event.set()

    def _reset_warn_budget(self) -> None:
        """
        Give a newly spawned daemon a full warning budget of its own.

        The budget is per daemon, not per provider: a daemon that spent it
        before crashing would otherwise have its replacement's startup failure -
        the one line worth reading - suppressed for the rest of the window.
        Whatever the old daemon had held back is reported on the way out rather
        than dropped.
        """
        if self._warns_suppressed:
            self.logger.warning(
                "PTP daemon: %d further problem line(s) from the previous daemon were "
                "suppressed; enable verbose logging to see them all",
                self._warns_suppressed,
            )
        self._warn_window_start = None
        self._warns_in_window = 0
        self._warns_suppressed = 0

    def _warn_line(self, line: str) -> None:
        """
        Warn about a daemon line that reads like a problem, at a bounded rate.

        The markers are broad by design, and "error" is ordinary vocabulary in
        clock telemetry (offset error, path delay error), so one matching line
        in the daemon's per-packet trace would fill a user's log with warnings.
        A burst still gets through - which is what a real one-shot daemon
        failure looks like - and the rest of the window is counted and reported
        once instead. Suppressed lines still reach the verbose log.

        :param line: The daemon output line that matched a problem marker.
        """
        now = time.monotonic()
        window_start = self._warn_window_start
        # None means no window is open yet, which is not the same as one that
        # started at monotonic zero: time.monotonic() counts from boot on Linux,
        # so a server started in the first minute of uptime would otherwise get
        # a first window shorter than the rest.
        if window_start is None or now - window_start >= PTP_DAEMON_WARN_WINDOW:
            if self._warns_suppressed:
                self.logger.warning(
                    "PTP daemon: %d further problem line(s) were suppressed over the last "
                    "%.0fs; enable verbose logging to see them all",
                    self._warns_suppressed,
                    PTP_DAEMON_WARN_WINDOW,
                )
            self._warn_window_start = now
            self._warns_in_window = 0
            self._warns_suppressed = 0
        if self._warns_in_window < PTP_DAEMON_WARN_BURST:
            self._warns_in_window += 1
            self.logger.warning("PTP daemon: %s", line)
            return
        self._warns_suppressed += 1
        self.logger.log(VERBOSE_LOG_LEVEL, "PTP daemon: %s", line)

    async def _monitor(self, daemon: AsyncProcess) -> None:
        """Watch the daemon process and restart it once if it crashes."""
        returncode = await daemon.wait()
        if self._stop_requested or self._process is not daemon:
            return
        self._process = None
        # Crash detected: drop readiness immediately so a session starting during
        # the restart window degrades consistently instead of attaching to a dead
        # clock (a restart re-sets it once the new daemon reports ready).
        if self._ready is not None:
            self._ready.clear()
        if self._compat and returncode == PTP_DAEMON_EXIT_NQPTP:
            # A real nqptp holds UDP 9000: keep PTP for the sender, and let the
            # receiver fall back to classic AirPlay.
            self._compat_blocked = True
            self.logger.warning(
                "UDP port 9000 is taken (is nqptp running on this host?) - "
                "AirPlay 2 receiving is unavailable"
            )
            await self._start(False)
            return
        runtime = time.monotonic() - self._started
        if runtime < 5:
            # immediate exit: UDP 319/320 already taken or missing privileges
            # (root or CAP_NET_BIND_SERVICE) - streams fall back to their
            # in-process timing engine, so playback keeps working.
            self.logger.warning(
                "PTP clock daemon could not start (exit code %s) - PTP timing is degraded. "
                "Multi-room sync of native AirPlay 2 players may drift; ensure UDP ports "
                "319/320 are free and run the server as root or grant cliairplay "
                "CAP_NET_BIND_SERVICE.",
                returncode,
            )
            return
        if not self._restarted:
            self._restarted = True
            self.logger.warning(
                "PTP clock daemon stopped unexpectedly (exit code %s) - restarting", returncode
            )
            await self._start(self._compat)
            return
        self.logger.warning(
            "PTP clock daemon stopped again (exit code %s) - giving up. "
            "PTP timing is degraded until the provider is reloaded.",
            returncode,
        )
