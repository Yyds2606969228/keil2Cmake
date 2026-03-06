from pathlib import Path

from openocd_mcp.parsers.svd_resolver import SVDResolver


def test_svd_resolver_load_and_decode(tmp_path: Path) -> None:
    svd = tmp_path / "test.svd"
    svd.write_text(
        """
<device>
  <peripherals>
    <peripheral>
      <name>TIM1</name>
      <baseAddress>0x40010000</baseAddress>
      <registers>
        <register>
          <name>CR1</name>
          <addressOffset>0x00</addressOffset>
          <fields>
            <field>
              <name>CEN</name>
              <bitOffset>0</bitOffset>
              <bitWidth>1</bitWidth>
            </field>
            <field>
              <name>DIR</name>
              <bitOffset>4</bitOffset>
              <bitWidth>1</bitWidth>
            </field>
          </fields>
        </register>
      </registers>
    </peripheral>
  </peripherals>
</device>
""".strip(),
        encoding="utf-8",
    )
    resolver = SVDResolver()
    resolver.load(str(svd))
    info = resolver.resolve("TIM1->CR1")
    assert info.address == 0x40010000
    decoded = resolver.decode_fields(info, 0x11)
    assert decoded["CEN"]["val"] == 1
    assert decoded["DIR"]["val"] == 1
    snippet = resolver.raw_xml_snippet("TIM1->CR1")
    assert snippet is not None
    fallback = resolver.resolve_best_effort("TIM1->CR1")
    assert fallback["address"] == 0x40010000
    assert fallback["raw_xml_snippet"] is not None
