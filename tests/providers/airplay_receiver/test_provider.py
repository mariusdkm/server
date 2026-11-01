"""Unit tests for the AirPlay Receiver provider (ports + daemon reconciliation)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from music_assistant_models.enums import EventType
from music_assistant_models.event import MassEvent

from music_assistant.providers.airplay_receiver import (
    AirPlayReceiverProvider,
    airplay_receiver_ports,
)
from music_assistant.providers.airplay_receiver.helpers import (
    ShairportSyncCapabilities,
    get_shairport_sync_capabilities,
)
from music_assistant.providers.airplay_receiver.mdns import ServiceRegistration
from music_assistant.providers.airplay_receiver.provider import _ReceiverDaemon

MODULE = "music_assistant.providers.airplay_receiver.provider"
HELPERS = "music_assistant.providers.airplay_receiver.helpers"

# --- Port derivation -----------------------------------------------------------


def test_airplay_receiver_ports_are_deterministic() -> None:
    """The port derivation must be stable across processes/restarts (unlike ``hash()``)."""
    # pinned expectations guard against accidental changes to the derivation:
    # the AirPlay provider relies on reproducing these ports from config alone
    assert airplay_receiver_ports("airplay_receiver", ["player-a", "player-b"]) == {
        "player-a": 7765,
        "player-b": 7158,
    }


def test_airplay_receiver_ports_stay_in_expected_range_and_unique() -> None:
    """Derived ports stay within the 7000-7999 AirPlay range and never collide."""
    ports = airplay_receiver_ports("airplay_receiver", [f"player{index}" for index in range(50)])
    assert all(7000 <= port <= 7999 for port in ports.values())
    assert len(set(ports.values())) == 50


def test_airplay_receiver_ports_deduplicates_player_ids() -> None:
    """A duplicated player id claims a single port instead of probing extra slots."""
    ports = airplay_receiver_ports("airplay_receiver", ["player-a", "player-a"])
    assert ports == {"player-a": 7765}


def test_airplay_receiver_ports_collision_probing_is_order_independent() -> None:
    """Colliding derivations probe deterministically regardless of the input order."""
    # player22 and player23 both derive base port 7525 for this instance id
    colliding = ["player22", "player23"]
    ports = airplay_receiver_ports("airplay_receiver", colliding)
    assert ports == {"player22": 7525, "player23": 7526}
    assert airplay_receiver_ports("airplay_receiver", list(reversed(colliding))) == ports


# --- Daemon reconciliation -----------------------------------------------------


@dataclass
class _ReconcileMocks:
    """The mocked collaborators of a reconcile-test provider."""

    mass: MagicMock
    start_receiver: MagicMock
    stop_receiver: AsyncMock


def _reconcile_provider(
    assigned: tuple[str, ...],
    registered: dict[str, str],
) -> tuple[AirPlayReceiverProvider, _ReconcileMocks]:
    """
    Build a bare provider with the real reconcile logic and mocked daemon control.

    :param assigned: The connected player ids the provider was loaded with.
    :param registered: Currently registered player ids mapped to their display name.
    """
    prov = AirPlayReceiverProvider.__new__(AirPlayReceiverProvider)
    prov.logger = MagicMock()
    prov.config = MagicMock()
    prov.mass = mass = MagicMock()
    prov._daemons = {}
    prov._failed_player_ids = set()
    prov._reconcile_lock = asyncio.Lock()
    prov._unload_called = False
    prov._unsubscribe = None
    prov._assigned_player_ids = assigned
    prov.get_config_value = MagicMock(return_value="player_mass")  # type: ignore[method-assign]

    def get_player(player_id: str) -> MagicMock | None:
        if player_id not in registered:
            return None
        player = MagicMock()
        player.player_id = player_id
        player.display_name = registered[player_id]
        return player

    mass.players.get_player.side_effect = get_player

    def start_receiver(player: MagicMock, airplay_name: str) -> None:
        # stop_called / active_player_id are spelled out: a bare MagicMock attribute is
        # truthy, which would trip the stopped-daemon guard and the deselect path
        prov._daemons[player.player_id] = MagicMock(
            player_id=player.player_id,
            airplay_name=airplay_name,
            stop_called=False,
            active_player_id=None,
        )

    start_mock = MagicMock(side_effect=start_receiver)
    stop_mock = AsyncMock()
    prov._start_receiver = start_mock  # type: ignore[method-assign]
    prov._stop_receiver = stop_mock  # type: ignore[method-assign]
    return prov, _ReconcileMocks(mass=mass, start_receiver=start_mock, stop_receiver=stop_mock)


async def test_reconcile_starts_daemon_when_assigned_player_registers() -> None:
    """A daemon starts only once its connected player has actually registered."""
    registered: dict[str, str] = {}
    prov, mocks = _reconcile_provider(("p1",), registered)

    # cold boot: the player has not registered yet, so nothing starts
    await prov._reconcile()
    mocks.start_receiver.assert_not_called()

    registered["p1"] = "Kitchen"
    await prov._reconcile()
    mocks.start_receiver.assert_called_once()
    assert mocks.start_receiver.call_args.args[1] == "Kitchen | Music Assistant"
    assert "p1" in prov._daemons


async def test_reconcile_restarts_daemon_on_advertised_name_drift() -> None:
    """A renamed player gets its daemon restarted with the new advertised name."""
    registered = {"p1": "Kitchen"}
    prov, mocks = _reconcile_provider(("p1",), registered)
    await prov._reconcile()
    old_daemon = prov._daemons["p1"]

    # a second pass without changes is a no-op
    await prov._reconcile()
    mocks.stop_receiver.assert_not_awaited()
    assert mocks.start_receiver.call_count == 1

    # a live session on the old daemon is released before the daemon is replaced
    old_daemon.active_player_id = "consumer"
    registered["p1"] = "Cellar"
    await prov._reconcile()
    mocks.stop_receiver.assert_awaited_once_with(old_daemon)
    assert prov._daemons["p1"].airplay_name == "Cellar | Music Assistant"
    mocks.mass.players.deselect_source.assert_called_once()
    assert mocks.mass.players.deselect_source.call_args.args[0] == "consumer"


async def test_reconcile_keeps_daemon_for_temporarily_unavailable_player() -> None:
    """A temporarily unregistered player keeps its running daemon (stable identity)."""
    registered = {"p1": "Kitchen"}
    prov, mocks = _reconcile_provider(("p1",), registered)
    await prov._reconcile()
    daemon = prov._daemons["p1"]

    registered.clear()
    await prov._reconcile()
    mocks.stop_receiver.assert_not_awaited()
    assert prov._daemons["p1"] is daemon


async def test_player_removed_event_stops_daemon() -> None:
    """A permanently removed player gets its daemon stopped and dropped."""
    registered = {"p1": "Kitchen"}
    prov, mocks = _reconcile_provider(("p1",), registered)
    await prov._reconcile()
    daemon = prov._daemons["p1"]

    await prov._on_player_event(MassEvent(event=EventType.PLAYER_REMOVED, object_id="p1"))
    mocks.stop_receiver.assert_awaited_once_with(daemon)
    assert not prov._daemons


async def test_player_added_event_triggers_reconcile() -> None:
    """A player registering (cold boot path) starts its daemon via the event handler."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})

    await prov._on_player_event(MassEvent(event=EventType.PLAYER_ADDED, object_id="p1"))
    mocks.start_receiver.assert_called_once()


async def test_loaded_in_mass_with_empty_connected_players_is_idle() -> None:
    """An empty connected-players selection loads the provider fully idle."""
    prov, mocks = _reconcile_provider((), {})

    await prov.loaded_in_mass()
    mocks.mass.subscribe.assert_not_called()
    mocks.start_receiver.assert_not_called()
    assert not prov._daemons


async def test_loaded_in_mass_subscribes_to_assigned_players_only() -> None:
    """Player events are only watched for the connected players."""
    prov, mocks = _reconcile_provider(("p1", "p2"), {})

    await prov.loaded_in_mass()
    mocks.mass.subscribe.assert_called_once()
    assert mocks.mass.subscribe.call_args.kwargs["id_filter"] == ("p1", "p2")


async def test_get_player_audio_sources_scopes_to_the_daemon_player() -> None:
    """Each receiver's source is bound to its own connected player only."""
    prov, _mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    await prov._reconcile()
    daemon = prov._daemons["p1"]

    assert prov.get_player_audio_sources("p1") == [daemon.audio_source]
    assert prov.get_player_audio_sources("p2") == []


async def test_unload_stops_all_daemons() -> None:
    """Unload stops every running daemon and stops watching player events."""
    registered = {"p1": "Kitchen", "p2": "Garage"}
    prov, mocks = _reconcile_provider(("p1", "p2"), registered)
    await prov._reconcile()
    unsubscribe = MagicMock()
    prov._unsubscribe = unsubscribe

    await prov.unload()
    unsubscribe.assert_called_once()
    assert mocks.stop_receiver.await_count == 2
    assert not prov._daemons


async def test_give_up_receiver_stops_only_the_failed_daemon() -> None:
    """A permanently failed receiver is dropped while the other receivers keep running."""
    prov, mocks = _reconcile_provider(("p1", "p2"), {"p1": "Kitchen", "p2": "Garage"})
    await prov._reconcile()
    daemon = prov._daemons["p1"]
    daemon.active_player_id = "consumer"

    await prov._give_up_receiver(daemon, "shairport-sync daemon failed to start multiple times.")

    assert "p1" not in prov._daemons
    assert "p2" in prov._daemons
    mocks.stop_receiver.assert_awaited_once_with(daemon)
    assert prov._failed_player_ids == {"p1"}
    mocks.mass.players.trigger_player_update.assert_called_with("p1")
    mocks.mass.players.deselect_source.assert_called_once()
    cast("MagicMock", prov.logger).warning.assert_called_once()


async def test_reconcile_skips_a_given_up_receiver() -> None:
    """A receiver that gave up permanently is not relaunched by an ordinary reconcile."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    prov._failed_player_ids = {"p1"}

    await prov._reconcile()

    mocks.start_receiver.assert_not_called()
    assert "p1" not in prov._daemons


async def test_player_added_gives_a_failed_receiver_a_fresh_start() -> None:
    """A player re-registering lifts the block and starts its receiver again."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    prov._failed_player_ids = {"p1"}

    await prov._on_player_event(MassEvent(event=EventType.PLAYER_ADDED, object_id="p1"))

    assert "p1" not in prov._failed_player_ids
    mocks.start_receiver.assert_called_once()
    assert "p1" in prov._daemons


async def test_give_up_on_a_replaced_receiver_is_a_noop() -> None:
    """A give-up landing after the receiver was replaced leaves the replacement running."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    await prov._reconcile()
    old_daemon = prov._daemons["p1"]
    replacement = MagicMock(player_id="p1", airplay_name="Kitchen | Music Assistant")
    prov._daemons["p1"] = replacement

    await prov._give_up_receiver(old_daemon, "boom")

    mocks.stop_receiver.assert_not_awaited()
    assert not prov._failed_player_ids
    assert prov._daemons["p1"] is replacement


async def test_runner_exit_retries_before_the_restart_budget_is_spent() -> None:
    """A daemon exit within the restart budget schedules a delayed restart."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    await prov._reconcile()
    daemon = prov._daemons["p1"]
    daemon.started = asyncio.Event()
    daemon.started.set()
    daemon.runner_error_count = 0

    prov._handle_runner_exit(daemon)

    assert daemon.runner_error_count == 1
    mocks.mass.call_later.assert_called_once_with(2, prov._setup_shairport_daemon, daemon)


async def test_runner_exit_gives_up_once_the_restart_budget_is_spent() -> None:
    """Exhausting the restart budget gives up this receiver instead of unloading."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    unload_with_error = MagicMock()
    prov.unload_with_error = unload_with_error  # type: ignore[method-assign]
    await prov._reconcile()
    daemon = prov._daemons["p1"]
    daemon.started = asyncio.Event()
    daemon.started.set()
    daemon.runner_error_count = 5

    prov._handle_runner_exit(daemon)
    # the give-up runs as a deferred task so it never stops its own runner task
    assert mocks.mass.create_task.call_args.kwargs == {"eager_start": False}
    await mocks.mass.create_task.call_args.args[0]

    assert "p1" not in prov._daemons
    assert prov._failed_player_ids == {"p1"}
    unload_with_error.assert_not_called()


async def test_runner_exit_without_any_output_unloads_the_provider() -> None:
    """A daemon that never produced any output is an environment-level provider failure."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    unload_with_error = MagicMock()
    prov.unload_with_error = unload_with_error  # type: ignore[method-assign]
    await prov._reconcile()
    daemon = prov._daemons["p1"]
    daemon.started = asyncio.Event()

    prov._handle_runner_exit(daemon)

    unload_with_error.assert_called_once()
    mocks.mass.create_task.assert_not_called()


async def test_runner_exit_after_a_deliberate_stop_does_nothing() -> None:
    """A deliberately stopped receiver never restarts, gives up or unloads."""
    prov, mocks = _reconcile_provider(("p1",), {"p1": "Kitchen"})
    unload_with_error = MagicMock()
    prov.unload_with_error = unload_with_error  # type: ignore[method-assign]
    await prov._reconcile()
    daemon = prov._daemons["p1"]
    daemon.started = asyncio.Event()
    daemon.stop_called = True

    prov._handle_runner_exit(daemon)

    unload_with_error.assert_not_called()
    mocks.mass.create_task.assert_not_called()
    mocks.mass.call_later.assert_not_called()


def test_a_fatal_log_line_marks_the_daemon_started() -> None:
    """A fatal-error log line still proves the binary runs, routing exits to the restart path."""
    prov, _mocks = _reconcile_provider(("p1",), {})
    daemon = MagicMock(started=asyncio.Event())

    prov._process_shairport_log_line(daemon, "fatal error: Could not bind any listening ports")

    assert daemon.started.is_set()


# --- AirPlay 2 -----------------------------------------------------------------


def _airplay2_provider(*, airplay2: bool, source_ip: str | None = None) -> AirPlayReceiverProvider:
    """Build a bare provider that can start receivers, in AirPlay 1 or AirPlay 2 mode."""
    prov = AirPlayReceiverProvider.__new__(AirPlayReceiverProvider)
    prov.logger = MagicMock()
    prov.mass = MagicMock()
    prov.mass.streams.get_source_ip = AsyncMock(return_value=source_ip)
    prov.config = MagicMock()
    prov.config.instance_id = "airplay_receiver"
    prov.config.name = "AirPlay Receiver"
    prov.manifest = MagicMock(domain="airplay_receiver")
    prov._daemons = {}
    prov._ports = {"media_player.kitchen": 7123}
    prov._audio_format = MagicMock()
    prov._ptp_daemon = MagicMock() if airplay2 else None
    prov._setup_shairport_daemon = MagicMock()  # type: ignore[method-assign]
    return prov


def _started(prov: AirPlayReceiverProvider, name: str = "Kitchen") -> _ReceiverDaemon:
    player = MagicMock(player_id="media_player.kitchen", display_name=name)
    prov._start_receiver(player, name)
    return prov._daemons["media_player.kitchen"]


async def test_airplay1_receiver_is_pinned_to_classic() -> None:
    """Without AirPlay 2, the receiver never turns itself into one (service_type "auto")."""
    prov = _airplay2_provider(airplay2=False)
    daemon = _started(prov)
    assert not daemon.smi_name
    assert daemon.advertiser is None
    assert await prov._get_service_lines(daemon) == '\tservice_type = "classic";\n'


async def test_airplay2_receiver_config() -> None:
    """An AirPlay 2 receiver is timed by the PTP daemon and advertised by the server."""
    prov = _airplay2_provider(airplay2=True, source_ip="192.168.1.5")
    daemon = _started(prov)
    lines = await prov._get_service_lines(daemon)
    assert lines.splitlines() == [
        '\tservice_type = "airplay2";',
        '\tmdns_backend = "metadata";',
        f'\tnqptp_shared_memory_interface_name = "{daemon.smi_name}";',
        f"\tairplay_device_id = 0x{daemon.device_id}L;",
        '\taddress = "192.168.1.5";',
    ]
    assert daemon.advertiser is not None

    # without a bound stream ip, it listens on every address
    unbound = _airplay2_provider(airplay2=True)
    assert "address" not in await unbound._get_service_lines(_started(unbound))


async def test_airplay2_identity_survives_a_rename() -> None:
    """IOS keeps knowing a renamed receiver: its device id follows the player, not the name."""
    prov = _airplay2_provider(airplay2=True)
    first = _started(prov, "Kitchen")
    renamed = _started(prov, "Cuisine")
    assert (renamed.smi_name, renamed.device_id) == (first.smi_name, first.device_id)

    # a valid nqptp interface name: "/" and at most 62 more characters
    assert first.smi_name.startswith("/ma-")
    assert len(first.smi_name) <= 63
    assert "/" not in first.smi_name[1:]
    # a locally administered, unicast MAC address
    assert len(first.device_id) == 12
    first_octet = int(first.device_id[:2], 16)
    assert first_octet & 0x02
    assert not first_octet & 0x01


async def test_service_registration_is_published() -> None:
    """A service from shairport-sync goes to the receiver's advertiser."""
    prov = _airplay2_provider(airplay2=True)
    daemon = _started(prov)
    advertiser = MagicMock()
    advertiser.publish = AsyncMock()
    daemon.advertiser = advertiser
    registration = ServiceRegistration("_airplay._tcp", "Kitchen", 7123, {})

    mass = MagicMock()
    prov.mass = mass
    prov._on_metadata_update(daemon, {"mdns_service": registration})

    mass.create_task.assert_called_once()
    await mass.create_task.call_args.args[0]
    advertiser.publish.assert_awaited_once_with(registration)


def _enable_provider(*, ready: bool, nqptp: bool) -> tuple[AirPlayReceiverProvider, MagicMock]:
    prov = AirPlayReceiverProvider.__new__(AirPlayReceiverProvider)
    prov.logger = MagicMock()
    prov.mass = MagicMock()
    prov.config = MagicMock()
    prov.config.instance_id = "airplay_receiver"
    prov._shairport_bin = "/bin/shairport-sync"
    ptp_daemon = MagicMock(nqptp_active=nqptp)
    ptp_daemon.acquire = AsyncMock()
    ptp_daemon.release = AsyncMock()
    ptp_daemon.wait_ready = AsyncMock(return_value=ready)
    return prov, ptp_daemon


async def _enable(
    capabilities: ShairportSyncCapabilities, *, ready: bool = True, nqptp: bool = True
) -> tuple[bool, AirPlayReceiverProvider, MagicMock]:
    prov, ptp_daemon = _enable_provider(ready=ready, nqptp=nqptp)
    with (
        patch(f"{MODULE}.get_shairport_sync_capabilities", AsyncMock(return_value=capabilities)),
        patch(f"{MODULE}.PtpDaemon.get", return_value=ptp_daemon),
    ):
        enabled = await prov._enable_airplay2()
    return enabled, prov, ptp_daemon


async def test_airplay2_enabled_when_everything_is_there() -> None:
    """AirPlay 2 runs with a capable binary and a daemon that stands in for nqptp."""
    enabled, prov, ptp_daemon = await _enable(ShairportSyncCapabilities(True, True))
    assert enabled
    assert prov._ptp_daemon is ptp_daemon
    ptp_daemon.acquire.assert_awaited_once_with("airplay_receiver", prov.logger, nqptp_compat=True)


async def test_airplay2_falls_back_without_a_capable_binary() -> None:
    """A shairport-sync without AirPlay 2 or the metadata backend stays AirPlay 1."""
    for capabilities in (
        ShairportSyncCapabilities(False, True),
        ShairportSyncCapabilities(True, False),
    ):
        enabled, prov, ptp_daemon = await _enable(capabilities)
        assert not enabled
        assert prov._ptp_daemon is None
        ptp_daemon.acquire.assert_not_awaited()


async def test_airplay2_falls_back_without_nqptp_compat() -> None:
    """A daemon that is not ready, or not standing in for nqptp, is released again."""
    for ready, nqptp in ((False, False), (True, False)):
        enabled, prov, ptp_daemon = await _enable(
            ShairportSyncCapabilities(True, True), ready=ready, nqptp=nqptp
        )
        assert not enabled
        assert prov._ptp_daemon is None
        ptp_daemon.release.assert_awaited_once_with("airplay_receiver")


async def test_shairport_sync_capabilities() -> None:
    """AirPlay 2 and the metadata mDNS backend are read from the binary itself."""
    usage = (
        b"Usage: shairport-sync [options...]\n"
        b"Available mDNS backends: \n    tinysvcmdns\n    metadata\n\n"
        b"Available audio backends: \n    pipe\n"
    )
    version = b"5.6.2-AirPlay2-smi11-OpenSSL-tinysvcmdns-pipe-metadata\n"
    outputs = {"-V": (0, version), "-h": (0, usage)}

    async def check_output(_binary: str, arg: str) -> tuple[int, bytes]:
        return outputs[arg]

    with patch(f"{HELPERS}.check_output", side_effect=check_output):
        assert (await get_shairport_sync_capabilities("ssp")).can_receive_airplay2

        outputs["-h"] = (0, usage.replace(b"    metadata\n", b""))
        assert not (await get_shairport_sync_capabilities("ssp")).mdns_metadata

        # "metadata" elsewhere in the usage text is not the backend
        outputs["-h"] = (0, usage.replace(b"    metadata\n", b"") + b"    metadata\n")
        assert not (await get_shairport_sync_capabilities("ssp")).mdns_metadata

        outputs["-V"] = (0, b"5.6.2-OpenSSL-tinysvcmdns-pipe-metadata\n")
        assert not (await get_shairport_sync_capabilities("ssp")).airplay2
