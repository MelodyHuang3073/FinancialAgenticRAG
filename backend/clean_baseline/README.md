# Naive RAG 對照組(標準 BM25,沒有任何系統處理)

這是對照組,不是消融實驗。目的:不用系統裡任何自己加上去的東西,量「原始 RAG」能答對多少題。這個資料夾**不 import 系統的任何程式碼**(`app.*`),也不讀任何環境變數開關。

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

## 狀態

2026-11-02 建立並離線測試;**還沒有執行**(沒有呼叫過 API)。要跑需要使用者下令。
