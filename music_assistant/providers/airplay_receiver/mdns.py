"""Publish the mDNS services of a shairport-sync running with the "metadata" mDNS backend."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zeroconf import NonUniqueNameException
from zeroconf.asyncio import AsyncServiceInfo

from music_assistant.helpers.util import get_ip_pton

if TYPE_CHECKING:
    from logging import Logger

    from music_assistant.mass import MusicAssistant


@dataclass
class ServiceRegistration:
    """One service as shairport-sync hands it over (an 'svrg' metadata item)."""

    service: str
    name: str
    port: int
    txt: dict[str, str]


def parse_service_payload(payload: str) -> ServiceRegistration | None:
    """
    Parse the payload of an 'svrg' item, or return None if it is malformed.

    :param payload: Lines of `service=`, `name=`, `port=` and `txt=<key=value>`.
    """
    fields: dict[str, str] = {}
    txt: dict[str, str] = {}
    for line in payload.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key == "txt":
            txt_key, _, txt_value = value.partition("=")
            if txt_key:
                txt[txt_key] = txt_value
        else:
            fields[key] = value
    try:
        port = int(fields["port"])
        service, name = fields["service"], fields["name"]
    except KeyError, ValueError:
        return None
    if not service.startswith("_") or not name or not 0 < port < 65536:
        return None
    return ServiceRegistration(service=service, name=name, port=port, txt=txt)


class ReceiverAdvertiser:
    """
    Publishes one receiver's services through Music Assistant's mDNS responder.

    Only the IPv4 stream address is advertised: the PTP daemon that times AirPlay 2
    reception follows senders over IPv4 only.
    """

    def __init__(self, mass: MusicAssistant, logger: Logger) -> None:
        """
        Initialize the advertiser.

        :param mass: The Music Assistant instance.
        :param logger: Logger of the receiver provider.
        """
        self.mass = mass
        self.logger = logger
        self._services: dict[str, AsyncServiceInfo] = {}

    async def publish(self, registration: ServiceRegistration) -> None:
        """
        Register a service, or update it when it is already registered.

        :param registration: The service as shairport-sync describes it.
        """
        zeroconf_type = f"{registration.service}.local."
        info = AsyncServiceInfo(
            zeroconf_type,
            name=f"{registration.name}.{zeroconf_type}",
            properties=registration.txt,
            addresses=[await get_ip_pton(self.mass.streams.publish_ip)],
            port=registration.port,
            server=f"{socket.gethostname()}.local",
        )
        aiozc = self.mass.discovery.aiozc
        try:
            if registration.service in self._services:
                await aiozc.async_update_service(info)
            else:
                await aiozc.async_register_service(info, strict=False)
        except NonUniqueNameException:
            self.logger.warning(
                "AirPlay receiver name '%s' is already in use on the network", registration.name
            )
            return
        self._services[registration.service] = info

    async def withdraw(self) -> None:
        """Unregister every service of the receiver."""
        services = list(self._services.values())
        self._services.clear()
        for info in services:
            await self.mass.discovery.aiozc.async_unregister_service(info)
