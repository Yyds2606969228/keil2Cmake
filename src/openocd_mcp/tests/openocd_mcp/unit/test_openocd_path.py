from openocd_mcp.core.openocd_path import resolve_openocd_binary


def test_resolve_openocd_binary_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENOCD_PATH", "C:/custom/openocd.exe")
    out = resolve_openocd_binary()
    assert out == "C:/custom/openocd.exe"


def test_resolve_openocd_binary_from_cfg(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "path.cfg"
    cfg.write_text(
        "[PATHS]\nopenocd_path = C:/Users/test/openocd.exe\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENOCD_PATH", raising=False)
    monkeypatch.setenv("OPENOCD_MCP_KEIL2CMAKE_CFG", str(cfg))
    out = resolve_openocd_binary()
    assert out == "C:/Users/test/openocd.exe"

