# Debug Runtime Integration

鏈洰褰曟壙杞戒粠 `openocd-mcp` 骞跺叆 `keil2cmake` 鐨勮皟璇曡繍琛屾椂鏂囨。銆?

## 缁勪欢瀹氫綅

璇ョ粍浠惰礋璐ｏ細
- OpenOCD 杩炴帴銆佹帶鍒朵笌鍒峰啓淇濇姢闂ㄧ
- 涓插彛缂撳啿銆佽Е鍙戜笌鍙岄€氶亾鍙栬瘉
- SVD / ELF 瑙ｆ瀽涓庣粨鏋勫寲杈撳嚭
- Python 浠诲姟杩愯鏃朵笌鑷姩鍖栭獙璇?
- 闈㈠悜 Agent 鐨?MCP 璋冭瘯鎺ュ彛

## 涓庝富浠撳垎宸?

- `keil2cmake` 涓讳綋璐熻矗宸ョ▼鐢熸垚銆佹瀯寤洪厤缃€佸伐鍏烽摼涓庝骇鐗╃鐞?
- `openocd-mcp` 缁勪欢璐熻矗杩愯鏈熻皟璇曘€佺幇鍦烘姄鍙栥€佹牴鍥犲垎鏋愭敮鎾戜笌鑷姩鍖栨墽琛?
- `keil2cmake.orchestrator` 璐熻矗鍏ㄥ眬鏂瑰悜缂栨帓銆佺姸鎬佹帹杩涗笌 handoff 鍐崇瓥

## 褰撳墠骞跺叆鍐呭

- 婧愮爜锛歚src/openocd_mcp/`
- 娴嬭瘯锛歚src/openocd_mcp/tests/openocd_mcp/`
- 鎶€鑳斤細`src/openocd_mcp/skills/`
- 鍗忚鏂囨。锛?
  - `api_contract.md`
  - `state_model.md`
  - `error_codes.md`

## 鍒嗗彂妯″瀷

褰撳墠鎺ㄨ崘閲囩敤 **鍙屽叆鍙?EXE 鍒嗗彂**锛?

- `Keil2Cmake.exe`锛氫富鍏ュ彛锛屾壙杞藉伐绋嬪伐鍏烽摼鑳藉姏涓庡叏灞€缂栨帓
- `openocd-mcp.exe`锛歁CP 鏈嶅姟鍏ュ彛锛屼緵涓婂眰瀹㈡埛绔互 `stdio` 鎷夎捣

璇ユā鍨嬩笅锛?

- 鍏ㄥ眬缂栨帓涓嶄細澶辨晥锛屼粛鐢?`keil2cmake.orchestrator` 璐熻矗
- MCP 杩愯鏃跺彧鏄 handoff 鐨勮兘鍔涘眰锛岃€屼笉鏄柊鐨勨€滀富鑴戔€?
- 榛樿涓嶄緷璧栧浐瀹?TCP 绔彛锛屽洜姝や笉浼氬紩鍏ュ吀鍨嬬鍙ｅ啿绐侀棶棰?

## 褰撳墠婕旇繘鐘舵€?

褰撳墠涓讳粨宸茬粡鍏峰锛?

- 鏂瑰悜缂栨帓灞?
- 璋冭瘯杩愯鏃?
- 宸ヤ欢涓€鑷存€х鐞?
- Agent 宸ヤ綔椤逛笌楠岃瘉鍥炵幆

鍚庣画婕旇繘閲嶇偣搴旀斁鍦細

- 鏇寸ǔ瀹氱殑鍙?EXE 鎵撳寘鑴氭湰
- MCP 瀹㈡埛绔帴鍏ョず渚?
- 涓诲叆鍙ｄ笌 MCP 鏈嶅姟鍏ュ彛鐨勫彂甯冩祦绋嬪浐鍖?

