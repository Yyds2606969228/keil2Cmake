import socket
from threading import Event, Thread

from openocd_mcp.transport.openocd_tcl.client import OpenOCDClient


def _fake_tcl_server(port: int, stop_event: Event) -> None:
    breakpoints: set[int] = set()
    watchpoints: set[int] = set()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        conn, _addr = server.accept()
        with conn:
            while not stop_event.is_set():
                buf = bytearray()
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        return
                    buf.extend(chunk)
                    if b"\x1a" in chunk:
                        break
                raw, _sep, _rest = bytes(buf).partition(b"\x1a")
                cmd = raw.decode("utf-8", errors="replace").strip()
                if cmd == "halt":
                    response = b"target halted\x1a"
                elif cmd == "reg pc":
                    response = b"pc (/32): 0x08000123\x1a"
                elif cmd.startswith("bp 0x"):
                    breakpoints.add(int(cmd.split()[1], 16))
                    response = b"ok:bp\x1a"
                elif cmd.startswith("rbp 0x"):
                    breakpoints.discard(int(cmd.split()[1], 16))
                    response = b"ok:rbp\x1a"
                elif cmd == "bp":
                    payload = " ".join(f"0x{x:08x}" for x in sorted(breakpoints))
                    response = payload.encode("utf-8") + b"\x1a"
                elif cmd.startswith("wp 0x"):
                    watchpoints.add(int(cmd.split()[1], 16))
                    response = b"ok:wp\x1a"
                elif cmd.startswith("rwp 0x"):
                    watchpoints.discard(int(cmd.split()[1], 16))
                    response = b"ok:rwp\x1a"
                elif cmd == "wp":
                    payload = " ".join(f"0x{x:08x}" for x in sorted(watchpoints))
                    response = payload.encode("utf-8") + b"\x1a"
                else:
                    response = f"ok:{cmd}".encode() + b"\x1a"
                conn.sendall(response)


def test_openocd_client_execute_and_control() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stop_event = Event()
    thread = Thread(target=_fake_tcl_server, args=(port, stop_event), daemon=True)
    thread.start()
    client = OpenOCDClient()
    client.connect(host="127.0.0.1", port=port)
    assert client.execute("mdw 0x0 1") == "ok:mdw 0x0 1"
    assert client.control_target("halt") == "target halted"
    assert client.get_pc() == "0x08000123"
    assert client.manage_breakpoint(point_type="bp", address=0x08000000, action="add").startswith("ok:")
    assert client.manage_breakpoint(point_type="wp", address=0x20000000, action="add").startswith("ok:")
    clear = client.clear_all_breakpoints()
    assert clear["verified"] is True
    stop_event.set()
    client.close()
