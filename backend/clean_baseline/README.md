# Naive RAG 對照組(標準 BM25,沒有任何系統處理)

這是疊加實驗的**第 0 步**(規劃見 `backend/experiment_results/EXPERIMENT_PLAN.md` 第 3 節),也是對照組,不是消融實驗。目的:不用系統裡任何自己加上去的東西,量「原始 RAG」能答對多少題。這個資料夾**不 import 系統的任何程式碼**(`app.*`),也不讀任何環境變數開關。

## 做了什麼

| 項目 | 內容 |
|---|---|
| 語料 | `build_corpus.py`:每份 PDF 用 PyMuPDF 原樣讀出文字,頁與頁用換行接起來,切成固定 1,000 字元、每次往後移 800(重疊 200)。**沒有**頁碼、id、type、company 欄位,沒有標頭,沒有表格處理,沒有清理空白 |
| 語料格式 | `{文件名稱: [chunk 字串, ...]}`。文件名稱只是用來決定這題用哪一份財報,不在 chunk 文字裡 |
| 用哪份財報 | 題庫的 `doc_name`;與 PDF 檔名只差標點或「dated」一個字,所以比對時只看英數字(`find_doc`) |
| 檢索 | 標準 Okapi BM25(`rank_bm25` 0.2.2 預設:k1=1.5、b=0.75、epsilon=0.25)。查詢與 chunk 都只做「轉小寫、用空白切詞」。沒有同義詞、沒有詞幹、沒有加權、沒有題型分類、沒有拆題 |
| 取幾個 | 前 16 個 chunk(對照用,與完整系統放進提示詞的個數相同),用空行接起來 |
| 提示詞 | 整段提示就是 `Context:` 加 chunk 原文,再加 `Question:` 加題目。沒有 system 訊息,沒有任何「怎麼回答」的指示 |
| 模型 | gpt-5-mini,呼叫一次,沒有設定溫度或推理強度。答案原樣存檔 |
| 沒有 | PoT、公式庫、Tri-Check、重試、規則式備援、會計慣例、題型指示 |

與 `experiment/raw-bm25-baseline` 分支的差別:那個分支用的是系統自己的檢索程式(只關掉加權)、語料每頁各自切塊並帶 `company`/`page_number`/`id`/`type` 欄位、提示詞有一句「依 context 回答,沒有就說沒有」。這個分支全部拿掉,那個分支的 300 題結果不能代表這裡的做法。

## 怎麼用

```
python backend/clean_baseline/build_corpus.py                  # 建語料(離線)
python backend/clean_baseline/run.py --set original --dry-run  # 印第一題的完整提示詞,不呼叫 API
python backend/clean_baseline/run.py --set original --shard 1/4 --out single_clean_original_1.json
cd backend/clean_baseline && python -m pytest -q               # 離線測試
```

`--shard i/n` 把題目平均分給 n 個程序;`--out` 每題存一次,中斷後重跑會從沒做過的題繼續。結果每題記錄:題目、標準答案、模型答案、檢索到的 chunk 序號(`retrieved`,只是序號,不進提示詞)、token 數、秒數。

## 結果(2026-11-20 重跑,gpt-5-mini、單檔、300 題各跑一次;依 EXPERIMENT_PLAN.md 第 9 節逐題讀完評分)

訓練集 150 題與獨立測試集 150 題分兩階段、各 4 個程序依序執行,用量與耗時分開記錄(階段時間窗在 `run_windows.json`)。300 題答案逐題讀過,每個非 C 的標籤都記下依據(決策表列號或第 9.4 節的規則),存在 `labels.json`,並通過 `experiment_results/grading_check.py`。判分者是 AI 助理,作者複核。

| | 正確 | 小誤差 | 部分 | 錯誤 | 答對率 |
|---|---|---|---|---|---|
| **Naive RAG 300 題** | **148** | 2 | 26 | 124 | **49.3%** |
| 其中訓練集 150 題 | 66 | 1 | 14 | 69 | 44.0% |
| 其中獨立測試集 150 題 | 82 | 1 | 12 | 55 | 54.7% |

- 依題型(正確題數,各 100 題):算指標題 20、領域題 51、說明題 77。
- 錯誤 124 題逐題讀過,分成兩類(結果檔 `w_kind` 欄):答案說找不到數字、請使用者補資料或反問(`lack`)**98 題**(訓練集 51、獨立測試集 47);給了答案但不對(`wrong`)**26 題**(訓練集 18、獨立測試集 8)。
- 用量(結果檔的 `tokens` 欄,兩個集合分開記):

| | 訓練集 | 獨立測試集 |
|---|---|---|
| 模型呼叫 | 150(0 次失敗) | 150(0 次失敗) |
| token 合計 | 677,245(輸入 550,514、輸出 126,731) | 641,434(輸入 539,461、輸出 101,973) |
| 每題 token | 4,514 | 4,276 |
| 每題耗時 平均 / 中位 / 最慢(秒) | 9.3 / 8.5 / 30.7 | 7.7 / 6.3 / 31.5 |
| 整段執行時間 | 421 秒 | 336 秒 |

- 兩個集合合計 1,318,679 token;每題耗時是 4 個程序同時跑時的單題時間。
- 單次執行。

檔案(在 `backend/experiment_results/stack_step0/`):`answers.json`(每題答案、檢索到的 chunk 序號、用量、`label` 與 `rule`)、`labels.json`(逐題標籤與依據)、`summary.json`、`summarize.py`(重算以上表格)、`run_windows.json`(兩個階段的起訖時間)。
