"""The address the web interface listens on.

Stored next to the model directory in the settings file, so the Setup menu of
either interface can change it. A change applies the next time the web server
starts: a running server cannot move to another socket under its own feet.

The host must be an IP address literal. A hostname would resolve to whatever
the resolver says at start-up, and "listen on 0.0.0.0" is a decision the user
should see spelled out, not one hidden behind a name.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass

from llmft.paths import read_settings, update_settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
HOST_KEY = "server_host"
PORT_KEY = "server_port"
MIN_PORT = 1
MAX_PORT = 65535


class ServerAddressError(ValueError):
    """Raised when a host or port cannot be listened on."""


@dataclass(frozen=True)
class ServerAddress:
    """An IP address and port pair for the web server."""

    host: str
    port: int

    @property
    def url(self) -> str:
        """The address as a URL, bracketing IPv6 hosts as URLs require."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def is_default(self) -> bool:
        """Whether this is the address used when nothing is stored."""
        return self == DEFAULT_ADDRESS

    @property
    def is_loopback(self) -> bool:
        """Whether only this machine can reach the server."""
        return ipaddress.ip_address(self.host).is_loopback


DEFAULT_ADDRESS = ServerAddress(DEFAULT_HOST, DEFAULT_PORT)


def server_address() -> ServerAddress:
    """Return the stored address, or the default for anything missing or damaged.

    Like the model directory, a damaged setting should not stop the
    application starting, so each half falls back on its own.
    """
    settings = read_settings()
    host, port = settings.get(HOST_KEY), settings.get(PORT_KEY)
    return ServerAddress(
        host if isinstance(host, str) and _is_ip_address(host) else DEFAULT_HOST,
        port if isinstance(port, int) and MIN_PORT <= port <= MAX_PORT else DEFAULT_PORT,
    )


def set_server_address(host: str, port: str | int) -> ServerAddress:
    """Validate ``host`` and ``port``, store them and return the new address.

    Empty values restore the defaults.
    """
    address = ServerAddress(validated_host(host), validated_port(port))
    update_settings({HOST_KEY: address.host, PORT_KEY: address.port})
    return address


def validated_host(raw: str) -> str:
    """Return ``raw`` as an IP address this machine can listen on, or raise.

    Binding to an ephemeral port proves the address belongs to a local
    interface; otherwise the mistake would only surface at the next start.
    """
    candidate = raw.strip() or DEFAULT_HOST
    if not _is_ip_address(candidate):
        raise ServerAddressError(f"{candidate!r} is not an IP address, e.g. 127.0.0.1 or 0.0.0.0.")
    host = str(ipaddress.ip_address(candidate))
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
    except OSError as error:
        raise ServerAddressError(f"{host} is not an address of this machine: {error.strerror or error}") from error
    return host


def validated_port(raw: str | int) -> int:
    """Return ``raw`` as a TCP port number, or raise."""
    text = str(raw).strip() or str(DEFAULT_PORT)
    port = int(text) if text.isascii() and text.isdigit() else -1
    if not MIN_PORT <= port <= MAX_PORT:
        raise ServerAddressError(f"{text!r} is not a port number between {MIN_PORT} and {MAX_PORT}.")
    return port


def _is_ip_address(text: str) -> bool:
    """Whether ``text`` is an IPv4 or IPv6 address literal."""
    try:
        ipaddress.ip_address(text)
    except ValueError:
        result = False
    else:
        result = True
    return result
