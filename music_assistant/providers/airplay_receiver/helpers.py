"""Helpers/utils for the AirPlay Receiver plugin."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass

from music_assistant.helpers.process import check_output


@dataclass(frozen=True)
class ShairportSyncCapabilities:
    """What a shairport-sync binary was built with."""

    airplay2: bool
    """AirPlay 2 (and the nqptp interface version Music Assistant serves)."""
    mdns_metadata: bool
    """The "metadata" mDNS backend, so Music Assistant can advertise its services."""

    @property
    def can_receive_airplay2(self) -> bool:
        """Return whether Music Assistant can run this binary as an AirPlay 2 receiver."""
        return self.airplay2 and self.mdns_metadata


async def get_shairport_sync_capabilities(shairport_bin: str) -> ShairportSyncCapabilities:
    """
    Return what a shairport-sync binary supports.

    :param shairport_bin: Path of the shairport-sync binary.
    """
    _, version = await check_output(shairport_bin, "-V")
    _, usage = await check_output(shairport_bin, "-h")
    backends = usage.decode(errors="ignore").partition("Available mDNS backends:")[2]
    return ShairportSyncCapabilities(
        airplay2=b"-AirPlay2" in version and b"-smi11" in version,
        mdns_metadata="metadata" in backends.split("\n\n", 1)[0].split(),
    )


async def get_shairport_sync_binary() -> str:
    """Find the shairport-sync binary (bundled or system-installed)."""

    async def check_shairport_sync(shairport_path: str) -> str | None:
        """Check if shairport-sync binary is valid."""
        try:
            returncode, _ = await check_output(shairport_path, "--version")
            if returncode == 0:
                return shairport_path
            return None
        except OSError:
            return None

    # First, check if bundled binary exists
    base_path = os.path.join(os.path.dirname(__file__), "bin")
    system = platform.system().lower().replace("darwin", "macos")
    architecture = platform.machine().lower()

    if shairport_binary := await check_shairport_sync(
        os.path.join(base_path, f"shairport-sync-{system}-{architecture}")
    ):
        return shairport_binary

    # If no bundled binary, check system PATH
    if system_binary := shutil.which("shairport-sync"):
        if shairport_binary := await check_shairport_sync(system_binary):
            return shairport_binary

    msg = (
        f"Unable to locate shairport-sync for {system}/{architecture}. "
        "Please install shairport-sync on your system or provide a bundled binary."
    )
    raise RuntimeError(msg)
