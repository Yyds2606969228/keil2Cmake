import socket

from openocd_mcp.transport.openocd_tcl.client import OpenOCDClient


class _TimeoutThenStopSocket:
    def __init__(self, stop_event) -> None:
        self._stop_event = stop_event
        self.calls = 0

    def recv(self, _size: int) -> bytes:
        self.calls += 1
        if self.calls == 1:
            raise socket.timeout()
        self._stop_event.set()
        return b""


def test_telnet_loop_keeps_running_on_read_timeout() -> None:
    client = OpenOCDClient()
    fake_sock = _TimeoutThenStopSocket(client._stop_event)  # noqa: SLF001
    client._telnet_socket = fake_sock  # noqa: SLF001

    client._telnet_loop()  # noqa: SLF001

    assert fake_sock.calls == 2
