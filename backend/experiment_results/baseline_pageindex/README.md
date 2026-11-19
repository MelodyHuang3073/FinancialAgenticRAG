# 對照 baseline:PageIndex(gpt-5-mini,訓練集 150 題)

方法:VectifyAI PageIndex 0.6.0(MIT,本機模式,無向量檢索,由語言模型沿文件目錄樹推理找頁);程式在 `backend/baselines/pageindex/`,環境與原始碼放在倉庫外。
每題只上傳對應的一份財報;每份財報建一次索引。只跑訓練集 150 題(獨立測試集待新題組定案後再跑)。
評分依 `EXPERIMENT_PLAN.md` 第 9 節,由 AI 助理逐題讀完答案判定;沒有獨立的第二位評審。

## 結果(C / S / P / W)

| 範圍 | 題數 | C | S | P | W |
|---|---|---|---|---|---|
| 訓練集 | 150 | **130** | 0 | 10 | 10 |

- 依題型:數值題 50 題全對;領域題 50 題 41 / 0 / 5 / 4;新穎題 50 題 39 / 0 / 5 / 6。
- 2 題(train-96、train-99)代理人沒在 max_turns 內完成,沒有答案,依決定維持答錯、不重試。
- 逐題標籤與判分依據在 `labels.json`(規則編號見 `grading_check.py`),非 C 題的說明在 `answers.json` 的 `note`。

## 用量
- 每題平均 29 秒、約 12.8 萬 token(148 題有答案);回答共 1,889 萬 token。
- 84 份財報的索引共 1,784 萬 token(每份平均約 21 萬)。
- 帳本總計 3,885 萬 token(`usage_ledger/train.jsonl`),比逐題與索引紀錄的加總多約 161 萬,來自失敗重試與連線錯誤重送的呼叫。帳本沒有題號,不能再細分到題。
- 試跑(正式執行前)的帳本在 `usage_ledger/trials.jsonl`,不計入結果。

## 檔案
`answers.json`(150 題答案、標籤、秒數、呼叫數、token)、`labels.json`、`summary.json`、`summarize.py`、`index_summary.json`、`progress.json`、`usage_ledger/`。
`answers_partial.json` 是執行中的合併檔(沒有標籤、不含 2 題錯誤紀錄),已由 `answers.json` 取代。
