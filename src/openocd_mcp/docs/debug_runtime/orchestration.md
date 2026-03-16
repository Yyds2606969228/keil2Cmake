# Global Orchestration Flow

褰撳墠鍏ㄥ眬缂栨帓宸叉寜 **鏂瑰悜缂栨帓锛坉irection orchestration锛?* 閲嶆瀯锛屼笉鍐嶆妸鎬诲伐浣滄祦瀹炵幇涓衡€滃浐瀹氶樁娈电姸鎬佹満 + 瑙勫垯琛ヤ竵鍣ㄢ€濄€?

鏂扮殑瀹氫綅鏄細

- 鐢?**鐩爣** 绾︽潫鏂瑰悜
- 鐢?**淇″彿** 鎵挎帴鐜板満浜嬪疄
- 鐢?**鍐崇瓥** 缁欏嚭涓嬩竴姝ユ柟鍚?
- 鐢?**Agent 宸ヤ綔椤?* 椹卞姩寮€鏀惧紡鎵ц
- 鐢?**楠岃瘉涓庡弽鎬?* 鎺у埗鍥炵幆

## 鐩爣

璇ョ紪鎺掑眰璐熻矗鎶婅蒋浠朵晶闂幆缁勭粐鎴愪互涓嬫柟鍚戞€у洖璺細

1. 鏄庣‘褰撳墠鎬荤洰鏍囦笌鎴愬姛鏉′欢
2. 鏍规嵁鏈€鏂颁俊鍙峰垽鏂綋鍓嶅叧娉ㄥ煙锛坋ngineering / build / artifact / debug / runtime / validation锛?
3. 缁欏嚭涓嬩竴姝ユ柟鍚戯紝鑰屼笉鏄啓姝荤粏绮掑害姝ラ
4. 鍦ㄥけ璐ユ椂鐢熸垚 Agent 宸ヤ綔椤癸紝璁?Agent 鑷富鎼滈泦淇℃伅銆佷慨鏀瑰伐浣滃尯骞剁珛鍗抽獙璇?
5. 鍦ㄦ垚鍔熴€佸け璐ャ€佸弽鎬濄€佷氦鎺ヤ箣闂翠繚鎸侀棴鐜彲杩芥函

## 鏍稿績妯″潡

- `src/keil2cmake/orchestrator/artifacts.py`
- `src/keil2cmake/orchestrator/models.py`
- `src/keil2cmake/orchestrator/state.py`
- `src/keil2cmake/orchestrator/planner.py`
- `src/keil2cmake/orchestrator/validation.py`
- `src/keil2cmake/orchestrator/workflow.py`

## 褰撳墠鑳藉姏杈圭晫

褰撳墠鐗堟湰鎻愪緵锛?

- 鍏ㄥ眬鐩爣銆佷俊鍙枫€佸喅绛栦笌宸ヤ綔椤瑰揩鐓?
- configure / build / debug preparation / triage / analysis / regression 鐨勬柟鍚戠紪鎺?
- 闈㈠悜 Agent 鐨勫紑鏀惧紡淇鍥炶矾锛堜笉渚濊禆鍥哄畾琛ヤ竵鑴氭湰锛?
- 宸ヤ欢娉ㄥ唽琛ㄤ笌涓€鑷存€ц瘎浼?
- 鐙珛楠岃瘉鎵ц灞傦紝鐢ㄤ簬缁熶竴 `configure/build` 鐨勫洖鐜獙璇?

褰撳墠鐗堟湰灏氭湭鎻愪緵锛?

- 璺ㄥ杞?session 鐨勬寔涔呭寲鎭㈠
- 杩愯鏈熷紓甯哥殑鑷姩鍖栨柟鍚戦噸瑙勫垝鍣?
- 浜烘満鍗忎綔寮忔壒鍑嗚妭鐐圭殑缁撴瀯鍖栧崗璁?

## 鏂瑰悜缂栨帓鍘熷垯

璇ュ眰閬靛惊 `AGENTS.MD` 涓殑 `Agent缂栨帓宸ヤ綔娴佹寚瀵煎師鍒檂锛?

- **鐩爣瀵煎悜**锛氱紪鎺掑眰鍏堝畾涔夊綋鍓嶇洰鏍囧拰鎴愬姛鏉′欢锛屽啀璁?Agent 鑷富鍐冲畾璺緞
- **鏉捐€﹀悎娴佺▼绾︽潫**锛氫繚鐣?`understand -> collect -> decide -> execute -> verify -> reflect -> handoff -> done` 鐨勬祦绋嬮鏋讹紝浣嗕笉鎶婇樁娈靛唴閮ㄥ姩浣滃啓姝?
- **宸ュ叿鍗宠兘鍔?*锛氭瀯寤恒€佽皟璇曘€佸彇璇併€侀獙璇佸彧鏄兘鍔涳紝缂栨帓灞傚彧鍐冲畾浣曟椂銆佷负浣曡皟鐢ㄥ畠浠?
- **鍙嶆€濅紭鍏堜簬纭紪鐮佽ˉ涓?*锛氬け璐ュ悗鍏堝舰鎴愬伐浣滈」涓庡€欓€夋枃浠讹紝鍐嶈 Agent 鍐冲畾鏈€灏忓姩浣?

## Agent 宸ヤ綔椤规ā鍨?

褰?`configure` 鎴?`build` 澶辫触鏃讹紝缂栨帓灞備細鐩存帴鐢熸垚 `AgentWorkItem`銆傚叾涓寘鍚細

- 褰撳墠 owner action锛堝 `configure` / `build`锛?
- 褰撳墠宸ヤ綔娴佺浉浣嶄笌鍏虫敞鍩?
- 褰撳墠鎬荤洰鏍?
- 鍘熷璇婃柇鏃ュ織
- 绾︽潫鏉′欢
- 鍊欓€夋枃浠跺揩鐓?
- 寤鸿鍔ㄤ綔锛堜粎浣滄柟鍚戝缓璁紝涓嶆槸鍥哄畾鑴氭湰锛?
- 楠岃瘉鍔ㄤ綔

鍥犳锛岄敊璇瘑鍒殑涓婚€昏緫宸蹭粠鈥滆鍒欏紡闂鍒嗙被鈥濊浆涓衡€滀繚鐣欏師濮嬭瘖鏂?-> Agent 鑷富鍒ゆ柇 -> 绔嬪嵆楠岃瘉鈥濄€?

## 褰撳墠鐘舵€佸瓧娈?

涓婂眰 Agent 褰撳墠搴旈噸鐐硅鍙栵細

- `workflow_phase`
- `phase_status`
- `focus_domain`
- `artifact_consistency`
- `current_goal`
- `current_signal`
- `active_work_item`
- `success_criteria`
- `constraints`
- `planned_actions`
- `completed_actions`
- `pending_action`
- `handoff_skill`
- `agent_iterations`

## 闈㈠悜 LLM 鐨勫叆鍙ｅ畾浣?

璇ョ紪鎺掑眰鐨勪富瑕佹秷璐硅€呬粛鐒舵槸 Skill 椹卞姩鐨?LLM / Agent锛岃€屼笉鏄汉宸?CLI銆?

鎺ㄨ崘鍏ュ彛锛?

- `src/openocd_mcp/skills/software-loop-orchestrator/SKILL.md`

鍏朵腑锛?

- `orchestrator` 鎻愪緵鐩爣銆佹柟鍚戙€佸伐浣滈」涓庡洖鐜姸鎬?
- Skill 鎻愪緵鍏ㄥ眬鏂瑰悜鍒ゆ柇鍘熷垯
- Tool 灞傚彧璐熻矗鎵ц鍘熷瓙鑳藉姏

## 涓庡弻鍏ュ彛 EXE 鍒嗗彂鐨勫叧绯?

閲囩敤鍙屽叆鍙?EXE 鍒嗗彂鏃讹紝杩欎竴灞備緷鐒舵湁鏁堬細

- `Keil2Cmake.exe` 浠嶇劧鏄富鍏ュ彛涓庢柟鍚戠紪鎺掓壙杞借€?
- `openocd-mcp.exe` 鏄 handoff 鐨?MCP 鏈嶅姟鍏ュ彛
- 缂栨帓灞傚彧鍐冲畾浣曟椂杩涘叆 `debug/runtime/validation` 鍩燂紝鑰屼笉鏄嚜宸卞彉鎴?MCP 鏈嶅姟鏈綋

鍥犳锛屽弻鍏ュ彛瑙ｅ喅鐨勬槸杩涚▼涓庡垎鍙戝舰鎬侊紝涓嶄細鍓婂急褰撳墠鏂瑰悜缂栨帓妯″瀷銆?

