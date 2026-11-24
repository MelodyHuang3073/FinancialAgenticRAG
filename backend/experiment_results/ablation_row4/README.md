# 消融列 4:拿掉 C5 系統檢索器(單檔 300 題)

拿掉 C5:所有計畫出的查詢以空白串成一個字串,用 `rank_bm25`(預設參數,小寫、空白切詞,只索引 `content`)搜一次、取 16 塊;沒有 8 條加權、配額、視窗 30;重試時每個查詢同樣用 `rank_bm25`。語料庫、答題提示詞、分類器與詞彙表、拆題器與公式庫、PoT、驗證與重試都照完整系統。規劃見上一層的 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) 第 4 節(列 4)與第 6 節,評分標準見第 9 節。比較對象是基準(`../baseline/`)。

## 怎麼跑的

| 項目 | 內容 |
|---|---|
| 日期 | 2026-11-24:訓練集 17:03 到 17:30(約 27 分鐘),獨立測試集 20:18 到 20:44(約 26 分鐘) |
| 程式 | `experiment/ablation` 的消融開關 commit,`ABLATE_ROW=4`。`config.json` 記錄的是啟動時分支頂端 `d18f008`;它的程式部分與消融開關 commit 相同(之後只加結果),結果重新 commit 後該 hash 不再存在 |
| 拿掉什麼 | C5:系統檢索器。檢索改成一個串接字串、`rank_bm25`、取 16 塊;規劃(拆題器與公式庫產生的查詢)不變 |
| 模型 | gpt-5-mini,每題只載入自己的財報,300 題(訓練集 150、獨立測試集 150)各跑一次 |
| 跑前檢查 | 程式與消融開關沒有改動,不另做冒煙測試;啟動檢查通過(`app/` 沒有未提交的修改);訓練集與獨立測試集各自啟動,啟動後逐一核對 4 個程序的指令列,確認集合是 `--set train` 或 `--set out`,互不混用 |
| 跑法 | 訓練集與獨立測試集分兩次啟動,各 2 個程序(共 4 個),啟動腳本是 `ablation_runner/launch.ps1` 的單一集合版本;`merge_check.py --row 4` 檢查 300 題齊全、沒有重複、沒有錯誤與備援合成答案、每筆記錄的列號等於 4 |
| 評分 | 依規劃第 9 節逐題讀完 300 題答案,每個非 C 的標籤記依據,通過 `grading_check.py`,並與基準、疊加第 0 到 4 步、兩個 Gemini、消融列 1、2a、2b 逐題比對(見下面「跨組比對」) |

## 結果

| | 正確 | 小誤差 | 部分 | 錯誤 | 答對率 |
|---|---|---|---|---|---|
| 300 題 | **256** | 2 | 19 | 23 | **85.3%** |
| 訓練集 150 題 | 130 | 0 | 9 | 11 | 86.7% |
| 獨立測試集 150 題 | 126 | 2 | 10 | 12 | 84.0% |

與基準(280 / 1 / 11 / 8)相比少 24 題正確(訓練集 145 → 130,少 15;獨立測試集 135 → 126,少 9),超過單次雜訊(約 7 題)。依題型(正確題數,各 100 題):算指標題 79、領域題 82、說明題 95。 錯誤 23 題裡,答「找不到、無法計算」(列 13)16 題、列 6 取錯數字 7 題。

### 沒有達到 C 的 44 題,依標籤與依據

| 標籤 | 依據 | 題數 | 說明 |
|---|---|---|---|
| W | 13 | 16 | 答找不到、無法計算、請貼資料:訓練集 13(Adobe What is the FY2017 operating cash…)、訓練集 22(Amcor What is Amcor's year end FY2020 ne…)、訓練集 71(Corning Based on the information provided…)、訓練集 77(Cvshealth Is CVS Health a capital-intensive…)、訓練集 82(Generalmills What is the FY2019 cash conversion…)、訓練集 100(Kraftheinz What is Kraft Heinz's FY2019 inven…)、訓練集 105(Mgmresorts What is the FY2018 - FY2020 3 year…)、訓練集 148(Walmart What is FY2018 days payable outsta…)、獨立測試集 6(Adobe Using Adobe's balance sheet and ca…)、獨立測試集 34(Bestbuy Compute Best Buy's cash conversion…)、獨立測試集 64(Cvshealth Is CVS Health's working capital po…)、獨立測試集 95(Mgmresorts Using FY2025 balance sheet data, d…)、獨立測試集 105(Netflix Using Netflix's balance sheet and…)、獨立測試集 106(Netflix What was Netflix's unadjusted EBIT…)、獨立測試集 123(Pepsico Compute PepsiCo's FY2025 inventory…)、獨立測試集 145(Walmart Calculate Walmart's fiscal 2025 DP…) |
| P | 9.4-4 | 6 | 結論對、關鍵內容(公式或口徑)不同:訓練集 3(3M Is 3M a capital-intensive business…)、訓練集 48(Americanwaterworks Does American Water Works have pos…)、獨立測試集 24(Amd What is AMD's quick ratio at the e…)、獨立測試集 60(Costco Looking at recent years, is Costco…)、獨立測試集 100(Microsoft What is Microsoft's quick ratio at…)、獨立測試集 149(Walmart Does Walmart have a reasonably hea…) |
| W | 6 | 7 | 用錯數字、科目或口徑(超過 5%):訓練集 49(Bestbuy In agreement with the information…)、訓練集 88(Johnson Roughly how many times has JnJ sol…)、訓練集 113(Netflix We want to calculate a financial m…)、獨立測試集 43(Cocacola Compute Coca-Cola's FY2025 return…)、獨立測試集 111(Nike What is Nike's fiscal 2025 fixed a…)、獨立測試集 125(Pepsico Using the cash flow statement and…、獨立測試集 54(Corning What share of Corning's total segm…)) |
| P | 6 | 5 | 輸入、口徑或年度名稱不同(差 5% 以內):訓練集 55(Bestbuy Was there any change in the number…)、訓練集 119(Paypal Does Paypal have positive working…)、獨立測試集 81(Kraftheinz If you divide Kraft Heinz's FY2025…)、獨立測試集 87(Lockheedmartin Calculate Lockheed Martin's FY2025…)、獨立測試集 134(Ultabeauty What was Ulta Beauty's asset turno…) |
| P | 15 | 3 | 只有定性結論、沒有標準答案要的數字:訓練集 146(Verizon Is Verizon a capital intensive bus…)、獨立測試集 32(Americanwaterworks What is American Water Works' work…)、獨立測試集 97(Mgmresorts Using FY2025 figures, is MGM Resor…) |
| P | 19 | 2 | 列舉題缺項:訓練集 121(Pepsico What are the geographies that Peps…)、訓練集 133(Pfizer What are three main companies acqu…) |
| S | 1 | 2 | 差最後一位:獨立測試集 129(Pfizer Did Pfizer's gross margin improve…)、獨立測試集 146(Walmart Averaged over fiscal 2023 to fisca…) |
| P | 9.4-10 | 1 | 是非題開頭的方向說反、其後內容與標準答案一致(降一級):訓練集 63(Boeing Does Boeing have an improving gros…) |
| P | 9.4-8 | 1 | 其餘正確、但含與證據互相矛盾的說法(降一級):訓練集 89(Johnson Is growth in JnJ's adjusted EPS ex…) |
| P | 18 | 1 | 原因題主要內容缺:獨立測試集 115(Nike Why did Nike's revenues decline in…) |

## 跨組比對(規劃 9.6)

- 標籤相同的題數:基準 259、疊加第 4 步 260、第 3 步 274、第 2 步 198、第 1 步 194、第 0 步 175、消融列 1 242、2a 253、2b 248、Gemini 3.1 251、Gemini 3.5 243。
- 基準非 C、這一列 C 的 7 題(訓練集 109、訓練集 135、訓練集 141、獨立測試集 7、獨立測試集 45、獨立測試集 92、獨立測試集 117):逐題對照後都是這一列的答案本身把標準答案要的內容寫對了,不是標準不一致。
- 基準 C、這一列非 C 的 31 題:逐題讀過,都是這一列的答案本身不同(找不到、結論相反、口徑或輸入不同、缺項),沒有標準不一致。
- 判斷依據的幾個決定(請注意):
  - 獨立測試集 24 與 100(速動比率):以流動資產減存貨計算(1.78、1.35),標準答案用現金、短期投資加應收帳款(1.78、1.16),判 P(9.4-4),與列 2a、2b 同判;獨立測試集 149(Walmart 速動比率 0.24 對 0.20)同為 P(9.4-4),與基準同判。
  - 獨立測試集 54(Corning 光通訊占比)答 42.0%(6,274 ÷ 14,948,分母只含可報導分部),標準答案 38%,判 W(第 6 列),與消融列 1 同判(2026-11-25 跨組重審)。
  - 獨立測試集 60(Costco 毛利率是否穩定)答 Yes,但毛利率以淨銷售額為分母,判 P(9.4-4),與基準同判。
  - 獨立測試集 32(American Water Works 營運資金)答「為赤字」但沒有數字,判 P(第 15 列),與疊加第 3 步同判;獨立測試集 97(MGM 資本密集)答 Yes,有資本支出占營收 6.1%,沒有資產對營收 2.36,判 P(第 15 列),與基準、疊加第 3、4 步同判。
  - 獨立測試集 81(Kraft Heinz 利息保障倍數)答 −4.93,標準答案為零(與訓練集同一慣例),判 P(第 6 列),與消融列 1、2a、2b、疊加第 3、4 步同判;獨立測試集 87(Lockheed DPO)答 15.61、標準答案 15.83,差 1.4%,判 P(第 6 列);獨立測試集 134(Ulta 資產周轉率)答 1.95、標準答案 1.91,判 P(第 6 列),與基準同判。
  - 獨立測試集 125(PepsiCo 盈餘保留率)用宣告股利 $7,718M 代替現金股利 $7,638M,答 0.06、標準答案 0.07,判 W(第 6 列);獨立測試集 111(Nike 固定資產週轉率)營收取 $48,763M(財報是 $46.3B),答 9.93、標準答案 9.42,判 W(第 6 列)。
  - 獨立測試集 129(Pfizer 毛利率)答 74.4% 與 +2.5 個百分點,標準答案 74.3% 與 +2.4,判 S(第 1 列),與消融列 1 同判。
  - 獨立測試集 66(CVS 總部)答 Woonsocket, Rhode Island 02895,沒有寫 One CVS Drive 街道,地點正確,判 C。
  - 第 6 列的 P 與 W 界線(2026-11-22 定):指標與公式對、只有輸入取錯,差 5% 以內判 P。

## 用量與耗時

用量取自 `usage_ledger/`(4 個程序)。訓練集與獨立測試集分兩次啟動,帳本檔屬於哪個集合由啟動紀錄決定,對應寫在 `summarize.py` 的 `SET_OF_PID`。

| 集合 | 呼叫次數 | token | 每題 token | 每題作答耗時 平均、中位、最慢 |
|---|---|---|---|---|
| 訓練集 | 273(150 + 123) | 2,660,787(輸入 2,396,813、輸出 263,974;作答 2,535,856、規劃器 124,931) | 17,739 | 20.8、20.0、44.3 秒 |
| 獨立測試集 | 285(150 + 135) | 2,684,527(輸入 2,436,186、輸出 248,341;作答 2,556,454、規劃器 128,073) | 17,897 | 20.6、19.7、45.9 秒 |
| 合計 | 558 | 5,345,314 | 17,818 | 20.7、19.9、45.9 秒 |

階段時間:訓練集 17:03 到 17:30,獨立測試集 20:18 到 20:44。拆題器(規劃器)呼叫訓練集 123 次、獨立測試集 135 次。基準是 5,012,899 token,這一列 5,345,314 token,約為基準的 107%。

## 檔案

| 檔案 | 內容 |
|---|---|
| `answers.json` | 300 題完整紀錄:題目、標準答案、模型答案、耗時、列號、標籤、依據 |
| `labels.json` | 逐題標籤與依據(給 `grading_check.py`) |
| `summary.json` | 統計與用量(含 `usage_by_set`:兩個集合分開的用量與耗時) |
| `summarize.py` | 由 `answers.json` 與 `usage_ledger/` 重算 `summary.json` |
| `config.json` | 啟動時記錄的列號、程式 commit、時間 |
| `usage_ledger/` | 本次 4 個程序的 LLM 呼叫帳本 |
