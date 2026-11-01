"""Helpers/utils for the AirPlay Receiver plugin."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass

from music_assistant.helpers.process import check_output

# the first release that hands its mDNS service over through the metadata pipe
MIN_SHAIRPORT_SYNC_VERSION = (5, 6, 3)


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
    """
    Return the path of the installed shairport-sync binary.

    :raises RuntimeError: When shairport-sync is not installed, too old, or built without the
        pipe backend or metadata support.
    """
    min_version = ".".join(str(part) for part in MIN_SHAIRPORT_SYNC_VERSION)
    if not (binary := shutil.which("shairport-sync")):
        msg = f"shairport-sync is not installed: the AirPlay Receiver needs version {min_version}+."
        raise RuntimeError(msg)
    try:
        returncode, output = await check_output(binary, "--version")
    except OSError as err:
        raise RuntimeError(f"Unable to run shairport-sync: {err}") from err
    version = output.decode(errors="ignore").strip()
    if returncode != 0 or not _is_supported_version(version):
        msg = (
            f"shairport-sync {version} can not be used: the AirPlay Receiver needs version "
            f"{min_version}+, with the pipe backend and metadata support."
        )
        raise RuntimeError(msg)
    return binary


def _is_supported_version(version: str) -> bool:
    """
    Return whether a shairport-sync build can run the receivers.

    :param version: The version string the binary prints, e.g.
        "5.6.3-FFmpeg-OpenSSL-stdout-pipe-metadata-sysconfdir:/etc".
    """
    if not (match := re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version)):
        return False
    release = tuple(int(part or 0) for part in match.groups())
    # the receivers need the audio pipe backend and metadata. An AirPlay 2 build is fine: the
    # configuration runs it as classic AirPlay, piping 16-bit 44.1 kHz audio
    capabilities = set(version.split("-"))
    return release >= MIN_SHAIRPORT_SYNC_VERSION and {"pipe", "metadata"} <= capabilities
