# OpenAI 換模型實驗的執行工具(gpt-5-nano、gpt-4o-mini)

規劃見 [EXPERIMENT_PLAN.md](../experiment_results/EXPERIMENT_PLAN.md) 第 8 節,評分規則在第 9 節。完整系統七個元件全開、不動任何系統程式,只把作答與拆題器的 LLM 換成指定模型:系統本來就從環境變數 `OPENAI_MODEL` 選模型(未設定時是基準的 gpt-5-mini),`gpt-5-*` 走推理模型的參數(不設 temperature、用 `max_completion_tokens`),`gpt-4o-mini` 走一般參數,所以**這個分支沒有改 `app/`**。

## 檔案

| 檔案 | 用途 |
|---|---|
| `launch.ps1` | 啟動 4 個背景程序(兩個題組各 2 份);`-Resume` 保留結果檔裡已有的紀錄、只跑沒有紀錄的題,並保留原本的開始時間。模型只能從 `-Model` 傳入,腳本設好 `OPENAI_MODEL`、移除 Google 金鑰與消融開關,並寫 `config.json`(模型、程式 commit、開始時間);`app/` 有未提交的修改就拒絕啟動 |
| `drop_failed.py` | 列出(`--apply` 才移除)結果檔裡失敗的紀錄:錯誤、空答案、fallback 形式的答案(額度用完時系統自己合成的文字)。移除後用 `launch.ps1 -Resume` 啟動就只重跑這些題(沒有 `-Resume` 會從頭開始並覆蓋結果檔) |
| `merge_check.py` | 合併 4 個結果檔並檢查:300 題齊全、沒有重複、沒有失敗的紀錄、`config.json` 的模型正確、用量帳本在開始時間之後沒有任何別的模型的呼叫;印出每個呼叫者的次數與 token |
| `failed.py` | 上面兩個工具共用的「失敗」定義 |
| `../tests/test_llm_model_swap.py` | 離線測試(不連網、不需金鑰):兩個模型的作答與拆題呼叫都用該模型與該家族能接受的參數、每次呼叫都以該模型記進用量帳本、合併檢查與失敗移除的行為 |

## 執行方式(開跑前先跟使用者確認,不自行啟動)

```powershell
# 1. 冒煙測試:每個分片只跑 1 題(共 4 題),看 .log 與 .out 正常、用量帳本有該模型
powershell -File backend\llm_runner\launch.ps1 -Model gpt-5-nano -RunDir C:\Users\y9205\financial_agentic_rag_runs\llm_gpt-5-nano_smoke -Python C:\Users\y9205\financial_agentic_rag\.venv\Scripts\python.exe -Limit 1
# 2. 正式:300 題
powershell -File backend\llm_runner\launch.ps1 -Model gpt-5-nano -RunDir C:\Users\y9205\financial_agentic_rag_runs\llm_gpt-5-nano -Python C:\Users\y9205\financial_agentic_rag\.venv\Scripts\python.exe
# 3. 4 個 .done 都出現後:失敗的題(呼叫失敗而由系統合成 fallback 答案的題)移掉再補跑,直到沒有失敗為止
python backend\llm_runner\drop_failed.py --run-dir <RunDir> --apply
# 補跑一定要加 -Resume;不加會把 300 題整個重跑並覆蓋結果檔(2026-11-12 gpt-4o-mini 發生過一次)
powershell -File backend\llm_runner\launch.ps1 -Model gpt-5-nano -RunDir <RunDir> -Python C:\Users\y9205\financial_agentic_rag\.venv\Scripts\python.exe -Resume
# 4. 合併檢查
python backend\llm_runner\merge_check.py --model gpt-5-nano --run-dir <RunDir> --out <RunDir>\merged.json
```

跑完後用量帳本(`backend/.llm_usage/`)只留這次執行之後的行,複製到結果資料夾的 `usage_ledger/`,再清空主帳本。

## 額度

OpenAI 的 mini 與 nano 小模型共用每個 UTC 日 10,000,000 token 的免費額度(UTC 00:00 = 台灣 08:00 重置)。一次 300 題約 440 萬到 550 萬 token(舊版執行的實測:gpt-4o-mini 約 442 萬、gpt-5-nano 約 549 萬;基準 gpt-5-mini 為 4,936,782),所以兩個模型**不放在同一天跑**:各自一個額度日,啟動前先看 `python -m app.agent.usage_tracker`(在 `backend/` 下)確認當天已用多少。
