from openocd_mcp.core.ring_buffer import LineRingBuffer, RecordRingBuffer


def test_ring_buffer_keeps_tail_by_capacity() -> None:
    buf = LineRingBuffer(capacity_bytes=20)
    buf.append("line-1")
    buf.append("line-2")
    buf.append("line-3")
    out = buf.tail(lines=10)
    assert "line-3" in out
    assert len(out) <= 3


def test_ring_buffer_keyword_filter() -> None:
    buf = LineRingBuffer(capacity_bytes=100)
    buf.append("hello world")
    buf.append("error: panic")
    buf.append("ok")
    out = buf.tail(lines=10, keyword="error")
    assert out == ["error: panic"]


def test_record_ring_buffer_overwrites_old_entries() -> None:
    buf = RecordRingBuffer(capacity_bytes=40)
    buf.append({"a": 1})
    buf.append({"b": 2})
    buf.append({"c": 3})
    snapshot = buf.snapshot()
    assert snapshot
    assert snapshot[-1] == {"c": 3}
