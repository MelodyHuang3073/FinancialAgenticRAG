# 換模型實驗:OpenAI(gpt-5-nano、gpt-4o-mini)

同一套流程、同一組單檔 300 題,只換 `OPENAI_MODEL`。基準是 gpt-5-mini(`../baseline/`)。每個模型一個資料夾:

| 資料夾 | 狀態 |
|---|---|
| `gpt_5_nano/` | 完成,訓練集 130、獨立測試集 140(基準 149、140) |
| `gpt_4o_mini/` | 尚未跑 |

分支 `experiment/llm-openai` 只放這些結果,不改程式碼。
