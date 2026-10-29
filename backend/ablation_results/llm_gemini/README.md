# 換模型實驗(Gemini)

老師問:換一個 LLM 會怎樣?這個分支把完整系統裡產生答案與拆題的 LLM 從 gpt-5-mini 換成 Google 的 Gemini Flash-Lite 兩個版本,其餘管線(分類器、檢索、PoT、Tri-Check)完全不變,單檔、300 題各跑一次,判分規則與基準和消融實驗相同。

| 模型 | 正確 | 小誤差 | 部分 | 錯誤 |
|---|---|---|---|---|
| 基準 gpt-5-mini | 289 | 2 | 5 | 4 |
| Gemini 3.1 Flash-Lite | 270 | 6 | 11 | 13 |
| Gemini 3.5 Flash-Lite | 263 | 11 | 11 | 15 |

- 各模型的逐題比較與說明在 `gemini_3_1_flash_lite/`、`gemini_3_5_flash_lite/`。
- 程式碼差異:`app/agent/gemini_call.py`(新增,遇到暫時性錯誤自動重試、記錄失敗)、`llm_client.py` 與 `decomposer.py` 的 Gemini 分支(答案提示放進系統指令)。沒有金鑰在 repo 裡,金鑰放各自 worktree 的 `backend/.env`(已被 git 忽略)。
- 免費額度每模型每天約 500 次請求,一次 300 題要分兩天,額度用完時系統會退回備援輸出,這些題已重跑,細節見各模型 README 的「備援答案與補跑」。
- 其他模型(Claude Haiku 5.5 需付費額度、OpenAI gpt-5-nano、gpt-4o-mini)還沒跑;Claude 的程式在 `experiment/llm-claude`。
- 出題與判分都是 Claude,測 Claude 系列模型時可能有偏向,論文裡要註明。
