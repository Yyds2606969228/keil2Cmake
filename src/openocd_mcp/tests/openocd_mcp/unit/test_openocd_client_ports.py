from openocd_mcp.transport.openocd_tcl.client import OpenOCDClient


def test_choose_ports_skips_busy_candidates(monkeypatch) -> None:
    client = OpenOCDClient()
    busy = {6666, 4444}

    def fake_is_port_free(port: int) -> bool:
        return port not in busy

    monkeypatch.setattr(client, "_is_port_free", fake_is_port_free)
    tcl, telnet = client._choose_ports(6666, 4444, 5)  # noqa: SLF001
    assert (tcl, telnet) == (6667, 4445)
