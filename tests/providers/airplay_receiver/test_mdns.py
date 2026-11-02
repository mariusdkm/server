"""Tests for publishing the services of a receiver running the "metadata" mDNS backend."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock

from zeroconf import NonUniqueNameException

from music_assistant.providers.airplay_receiver.mdns import (
    ReceiverAdvertiser,
    ServiceRegistration,
    parse_service_payload,
)

# As shairport-sync sends it (abridged TXT records)
AIRPLAY_PAYLOAD = (
    "service=_airplay._tcp\nname=Kitchen\nport=7123\n"
    "txt=deviceid=f6:70:0c:8c:fb:ea\ntxt=features=0x401C4A00,0x5C340\n"
    "txt=model=ShairportSync\ntxt=pk=9c05\n"
)


def test_parse_service_payload() -> None:
    """Every field and TXT record is taken over, in order."""
    registration = parse_service_payload(AIRPLAY_PAYLOAD)
    assert registration == ServiceRegistration(
        service="_airplay._tcp",
        name="Kitchen",
        port=7123,
        txt={
            "deviceid": "f6:70:0c:8c:fb:ea",
            "features": "0x401C4A00,0x5C340",
            "model": "ShairportSync",
            "pk": "9c05",
        },
    )
    assert registration is not None
    assert list(registration.txt) == ["deviceid", "features", "model", "pk"]


def test_parse_service_payload_keeps_equals_in_names_and_values() -> None:
    """Only the first '=' of a line separates; names and TXT values may contain more."""
    registration = parse_service_payload("service=_raop._tcp\nname=AB@A=B\nport=7000\ntxt=x=a=b\n")
    assert registration is not None
    assert registration.name == "AB@A=B"
    assert registration.txt == {"x": "a=b"}


def test_parse_service_payload_rejects_malformed() -> None:
    """A payload that cannot be advertised is dropped."""
    for payload in (
        "",
        "name=Kitchen\nport=7000\n",
        "service=_airplay._tcp\nport=7000\n",
        "service=_airplay._tcp\nname=Kitchen\nport=x\n",
        "service=_airplay._tcp\nname=Kitchen\nport=0\n",
        "service=airplay\nname=Kitchen\nport=7000\n",
    ):
        assert parse_service_payload(payload) is None


def _advertiser() -> tuple[ReceiverAdvertiser, MagicMock]:
    mass = MagicMock()
    mass.streams.publish_ip = "192.168.1.5"
    aiozc = mass.discovery.aiozc
    aiozc.async_register_service = AsyncMock()
    aiozc.async_update_service = AsyncMock()
    aiozc.async_unregister_service = AsyncMock()
    return ReceiverAdvertiser(mass, MagicMock()), aiozc


async def test_publish_registers_then_updates() -> None:
    """The first registration of a service registers it, later ones update it."""
    advertiser, aiozc = _advertiser()
    registration = parse_service_payload(AIRPLAY_PAYLOAD)
    assert registration is not None

    await advertiser.publish(registration)
    aiozc.async_register_service.assert_awaited_once()
    info = aiozc.async_register_service.await_args.args[0]
    assert info.type == "_airplay._tcp.local."
    assert info.name == "Kitchen._airplay._tcp.local."
    assert info.port == 7123
    assert info.server == f"{socket.gethostname()}.local"
    assert info.properties[b"features"] == b"0x401C4A00,0x5C340"
    # IPv4 only: the PTP daemon follows senders over IPv4
    assert info.parsed_addresses() == ["192.168.1.5"]

    registration.txt["flags"] = "0x804"
    await advertiser.publish(registration)
    aiozc.async_update_service.assert_awaited_once()
    assert aiozc.async_update_service.await_args.args[0].properties[b"flags"] == b"0x804"


async def test_withdraw_unregisters_every_service() -> None:
    """Both services go when the receiver stops."""
    advertiser, aiozc = _advertiser()
    for service in ("_airplay._tcp", "_raop._tcp"):
        await advertiser.publish(ServiceRegistration(service, "Kitchen", 7123, {}))

    await advertiser.withdraw()
    assert aiozc.async_unregister_service.await_count == 2
    await advertiser.withdraw()
    assert aiozc.async_unregister_service.await_count == 2


async def test_name_in_use_is_not_tracked() -> None:
    """A name taken on the network is reported and not unregistered later."""
    advertiser, aiozc = _advertiser()
    aiozc.async_register_service.side_effect = NonUniqueNameException
    await advertiser.publish(ServiceRegistration("_airplay._tcp", "Kitchen", 7123, {}))

    await advertiser.withdraw()
    aiozc.async_unregister_service.assert_not_awaited()
