# 消融列 3:拿掉 C4 拆題器與公式庫(單檔 300 題)

拿掉 C4:不拆題、不呼叫拆題器、公式庫關閉(`detect_formula` 回傳 None)、不補隱含的前一年;數值題只搜問題原文,說明題搜分類器的查詢句加問題原文。語料庫、答題提示詞、分類器與詞彙表、系統檢索器、驗證重試都照完整系統。規劃見上一層的 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) 第 4 節(列 3)與第 6 節,評分標準見第 9 節。比較對象是基準(`../baseline/`)。

## 怎麼跑的

| 項目 | 內容 |
|---|---|
| 日期 | 2026-11-24:訓練集 16:45 到 17:02(約 17 分鐘),獨立測試集 20:18 到 20:35(約 17 分鐘) |
| 程式 | `experiment/ablation` 的消融開關 commit,`ABLATE_ROW=3`。`config.json` 記錄的是啟動時分支頂端 `d18f008`;它的程式部分與消融開關 commit 相同(之後只加結果),結果重新 commit 後該 hash 不再存在 |
| 拿掉什麼 | C4:拆題器與公式庫。數值題的查詢只剩問題原文(1 個查詢),說明題是分類器查詢句加問題原文;沒有公式變數查詢、沒有模型拆出的子問句、沒有敘述主題查詢、沒有隱含前一年;沒有任何規劃器呼叫 |
| 模型 | gpt-5-mini,每題只載入自己的財報,300 題(訓練集 150、獨立測試集 150)各跑一次 |
| 跑前檢查 | 程式與消融開關沒有改動,不另做冒煙測試;啟動檢查通過(`app/` 沒有未提交的修改);訓練集與獨立測試集各自啟動,啟動後逐一核對 4 個程序的指令列,確認集合是 `--set train` 或 `--set out`,互不混用 |
| 跑法 | 訓練集與獨立測試集分兩次啟動,各 2 個程序(共 4 個),啟動腳本是 `ablation_runner/launch.ps1` 的單一集合版本;`merge_check.py --row 3` 檢查 300 題齊全、沒有重複、沒有錯誤與備援合成答案、每筆記錄的列號等於 3 |
| 評分 | 依規劃第 9 節逐題讀完 300 題答案,每個非 C 的標籤記依據,通過 `grading_check.py`,並與基準、疊加第 0 到 4 步、兩個 Gemini、消融列 1、2a、2b 逐題比對(見下面「跨組比對」) |

## 結果

| | 正確 | 小誤差 | 部分 | 錯誤 | 答對率 |
|---|---|---|---|---|---|
| 300 題 | **211** | 1 | 19 | 69 | **70.3%** |
| 訓練集 150 題 | 97 | 0 | 14 | 39 | 64.7% |
| 獨立測試集 150 題 | 114 | 1 | 5 | 30 | 76.0% |

與基準(280 / 1 / 11 / 8)相比少 69 題正確(訓練集 145 → 97,少 48;獨立測試集 135 → 114,少 21),超過單次雜訊(約 7 題)。依題型(正確題數,各 100 題):算指標題 55、領域題 69、說明題 87。 錯誤裡答「找不到、無法計算」(列 13)有 57 題(訓練集 31、獨立測試集 26)之外還有列 6 的取錯數字 10 題:沒有公式變數查詢與模型拆題,檢索常取不到算指標題需要的資產負債表、現金流量表欄位。

### 沒有達到 C 的 89 題,依標籤與依據

| 標籤 | 依據 | 題數 | 說明 |
|---|---|---|---|
| W | 13 | 57 | 答找不到、無法計算、請貼資料:訓練集 6(3M Does 3M have a reasonably healthy…)、訓練集 9(Activisionblizzard What is the FY2019 fixed asset tur…)、訓練集 11(Adobe You are an investment banker and y…)、訓練集 13(Adobe What is the FY2017 operating cash…)、訓練集 14(Adobe Does Adobe have an improving opera…)、訓練集 15(Adobe Does Adobe have an improving Free…)、訓練集 17(Aes Roughly how many times has AES Cor…)、訓練集 19(Amazon What is Amazon's FY2017 days payab…)、訓練集 24(Amcor Has AMCOR's quick ratio improved o…)、訓練集 31(Amd Answer the following question as i…)、訓練集 32(Amd Does AMD have a reasonably healthy…)、訓練集 47(Americanwaterworks Basing your judgments off of the c…)、訓練集 68(Cocacola What is the FY2017 return on asset…)、訓練集 70(Cocacola What is Coca Cola's FY2022 dividen…)、訓練集 72(Corning Taking into account the informatio…)、訓練集 76(Cvshealth What is the FY2018 fixed asset tur…)、訓練集 77(Cvshealth Is CVS Health a capital-intensive…)、訓練集 82(Generalmills What is the FY2019 cash conversion…)、訓練集 85(Generalmills We want to calculate a financial m…)、訓練集 88(Johnson Roughly how many times has JnJ sol…)、訓練集 95(Jpmorgan Which of JPM's business segments h…)、訓練集 100(Kraftheinz What is Kraft Heinz's FY2019 inven…)、訓練集 104(Mgmresorts Basing your judgments off of the b…)、訓練集 105(Mgmresorts What is the FY2018 - FY2020 3 year…)、訓練集 113(Netflix We want to calculate a financial m…)、訓練集 115(Nike We need to calculate a reasonable…)、訓練集 117(Nike When primarily referencing the inc…)、訓練集 124(Pepsico What is the FY2022 unadjusted EBIT…)、訓練集 125(Pepsico What is the FY2022 unadjusted EBIT…)、訓練集 148(Walmart What is FY2018 days payable outsta…)、訓練集 150(Walmart What is the FY2018 - FY2020 3 year…)、獨立測試集 6(Adobe Using Adobe's balance sheet and ca…)、獨立測試集 7(Adobe What is Adobe's quick ratio at the…)、獨立測試集 14(Amazon Calculate Amazon's FY2025 asset tu…)、獨立測試集 21(Amd Answer the following question as i…)、獨立測試集 22(Amd What is AMD's FY2025 return on equ…)、獨立測試集 23(Amd Over the two years from FY2023 to…)、獨立測試集 24(Amd What is AMD's quick ratio at the e…)、獨立測試集 34(Bestbuy Compute Best Buy's cash conversion…)、獨立測試集 44(Cocacola Using the cash flow statement and…)、獨立測試集 57(Costco Using the balance sheet and income…)、獨立測試集 58(Costco Averaged over fiscal 2023 through…)、獨立測試集 64(Cvshealth Is CVS Health's working capital po…)、獨立測試集 68(Footlocker Did Foot Locker distribute cash di…)、獨立測試集 81(Kraftheinz If you divide Kraft Heinz's FY2025…)、獨立測試集 87(Lockheedmartin Calculate Lockheed Martin's FY2025…)、獨立測試集 97(Mgmresorts Using FY2025 figures, is MGM Resor…)、獨立測試集 99(Microsoft What is Microsoft's fiscal 2025 re…)、獨立測試集 100(Microsoft What is Microsoft's quick ratio at…)、獨立測試集 105(Netflix Using Netflix's balance sheet and…)、獨立測試集 108(Netflix Using FY2025 figures, is Netflix a…)、獨立測試集 123(Pepsico Compute PepsiCo's FY2025 inventory…)、獨立測試集 125(Pepsico Using the cash flow statement and…)、獨立測試集 129(Pfizer Did Pfizer's gross margin improve…)、獨立測試集 140(Verizon Using the balance sheet and income…)、獨立測試集 145(Walmart Calculate Walmart's fiscal 2025 DP…)、獨立測試集 149(Walmart Does Walmart have a reasonably hea…) |
| W | 6 | 10 | 用錯數字、科目或口徑(超過 5%):訓練集 22(Amcor What is Amcor's year end FY2020 ne…)、訓練集 44(Americanexpress What was the largest liability in…)、訓練集 71(Corning Based on the information provided…)、訓練集 90(Johnson How did JnJ's US sales growth comp…)、訓練集 96(Jpmorgan If JPM went bankrupted by the end…)、訓練集 101(Lockheedmartin We need to calculate a reasonable…)、訓練集 135(Pfizer For Pfizer, which geographic regio…)、獨立測試集 43(Cocacola Compute Coca-Cola's FY2025 return…)、獨立測試集 112(Nike From the income statement and cash…、獨立測試集 54(Corning What share of Corning's total segm…)) |
| P | 9.4-4 | 5 | 結論對、關鍵內容(公式或口徑)不同:訓練集 3(3M Is 3M a capital-intensive business…)、訓練集 61(Boeing Are there any product categories /…)、訓練集 98(Jpmorgan In 2022 Q2, which of JPM's busines…)、訓練集 147(Verizon Has Verizon increased its debt on…)、獨立測試集 60(Costco Looking at recent years, is Costco…) |
| P | 6 | 5 | 輸入、口徑或年度名稱不同(差 5% 以內):訓練集 55(Bestbuy Was there any change in the number…)、訓練集 74(Corning Does Corning have positive working…)、訓練集 108(Mgmresorts What was MGM's interest coverage r…)、訓練集 119(Paypal Does Paypal have positive working…)、獨立測試集 147(Walmart What was Walmart's cost of sales a…) |
| P | 15 | 4 | 只有定性結論、沒有標準答案要的數字:訓練集 48(Americanwaterworks Does American Water Works have pos…)、訓練集 112(Microsoft Has Microsoft increased its debt o…)、訓練集 145(Verizon Does Verizon have a reasonably hea…)、訓練集 146(Verizon Is Verizon a capital intensive bus…) |
| P | 19 | 2 | 列舉題缺項:訓練集 40(Americanexpress What are the geographies that Amer…)、訓練集 121(Pepsico What are the geographies that Peps…) |
| W | 14 | 2 | 是非題結論和標準答案相反:訓練集 63(Boeing Does Boeing have an improving gros…)、獨立測試集 52(Corning On FY2025 revenue growth, is Corni…) |
| P | 9.4-8 | 1 | 其餘正確、但含與證據互相矛盾的說法(降一級):獨立測試集 85(Kraftheinz What are Kraft Heinz's reportable…) |
| P | 9.4-1 | 1 | 原因題只講次要原因、沒提到主因:獨立測試集 92(Lockheedmartin What drove the increase in Lockhee…) |
| P | 18 | 1 | 原因題主要內容缺:獨立測試集 115(Nike Why did Nike's revenues decline in…) |
| S | 1 | 1 | 差最後一位:獨立測試集 146(Walmart Averaged over fiscal 2023 to fisca…) |

## 跨組比對(規劃 9.6)

- 標籤相同的題數:基準 216、疊加第 4 步 219、第 3 步 235、第 2 步 228、第 1 步 217、第 0 步 210、消融列 1 218、2a 209、2b 210、Gemini 3.1 217、Gemini 3.5 211。
- 基準非 C、這一列 C 的 6 題(訓練集 109、訓練集 141、獨立測試集 45、獨立測試集 106、獨立測試集 117、獨立測試集 134):逐題對照後都是這一列的答案本身把標準答案要的內容寫對了,不是標準不一致。
- 基準 C、這一列非 C 的 75 題:逐題讀過,都是這一列的答案本身不同(找不到、結論相反、口徑或輸入不同、缺項),沒有標準不一致。
- 判斷依據的幾個決定(請注意):
  - 獨立測試集 106(Netflix EBITDA 利潤率):這一列答 30.2%,用現金流量表的折舊與攤銷 $333M 加營業利益,與標準答案相同,判 C;其他組都把內容攤銷也算進去(66.6%),判 W(第 6 列)。
  - 獨立測試集 52(Corning 是否高成長):答 No,同時寫出營收成長 19.14%;標準答案 Yes,結論相反,判 W(第 14 列)。
  - 獨立測試集 49(Corning 資產報酬率)答 5.93%(含非控制權益的淨利),與基準、消融列 1 同判 C。
  - 獨立測試集 54(Corning 光通訊占比)答 42.0%(6,274 ÷ 14,948,分母只含可報導分部),標準答案 38%,判 W(第 6 列),與消融列 1 同判(2026-11-25 跨組重審)。
  - 獨立測試集 60(Costco 毛利率是否穩定)答 Yes,但毛利率以淨銷售額為分母(10.57% 到 11.12%),標準答案 12.6% 與 12.8%,判 P(9.4-4),與基準、疊加第 4 步同判。
  - 獨立測試集 85(Kraft Heinz 可報導分部):答三個可報導分部(把 Emerging Markets 當成分部),又引用「兩個可報導分部」,判 P(9.4-8),與基準、消融列 1 同判。
  - 獨立測試集 92(Lockheed 產品銷售成長原因):只提 F-35、F-16 與 Aeronautics 的 $1.6B,沒有標準答案要的 Missiles and Fire Control $1.7B,判 P(9.4-1),與基準同判。
  - 獨立測試集 147(Walmart 銷貨成本占比):答 75.9%(占淨銷售額),標準答案 75.1%(占總營收),分母不同,判 P(第 6 列),與疊加第 3、4 步同判。
  - 獨立測試集 115(Nike 營收下降):答產品與通路,沒有北美、EMEA、大中華區,判 P(第 18 列),與基準同判。
  - 速動比率、營運資金類的是非題,答「No,但無法判斷」(例如獨立測試集 149、64)一律判 W(第 13 列)。
  - 第 6 列的 P 與 W 界線(2026-11-22 定):指標與公式對、只有輸入取錯,差 5% 以內判 P。

## 用量與耗時

用量取自 `usage_ledger/`(4 個程序)。訓練集與獨立測試集分兩次啟動,帳本檔屬於哪個集合由啟動紀錄決定,對應寫在 `summarize.py` 的 `SET_OF_PID`。

| 集合 | 呼叫次數 | token | 每題 token | 每題作答耗時 平均、中位、最慢 |
|---|---|---|---|---|
| 訓練集 | 150 | 2,046,570(輸入 1,873,067、輸出 173,503) | 13,644 | 13.4、12.5、28.9 秒 |
| 獨立測試集 | 150 | 2,074,749(輸入 1,917,407、輸出 157,342) | 13,832 | 12.7、12.1、30.3 秒 |
| 合計 | 300 | 4,121,319 | 13,738 | 13.0、12.3、30.3 秒 |

階段時間:訓練集 16:45 到 17:02,獨立測試集 20:18 到 20:35。這一列沒有規劃器呼叫,token 約為基準的 82%(基準 5,012,899 token,這一列 4,121,319)。

## 檔案

| 檔案 | 內容 |
|---|---|
| `answers.json` | 300 題完整紀錄:題目、標準答案、模型答案、耗時、列號、標籤、依據 |
| `labels.json` | 逐題標籤與依據(給 `grading_check.py`) |
| `summary.json` | 統計與用量(含 `usage_by_set`:兩個集合分開的用量與耗時) |
| `summarize.py` | 由 `answers.json` 與 `usage_ledger/` 重算 `summary.json` |
| `config.json` | 啟動時記錄的列號、程式 commit、時間 |
| `usage_ledger/` | 本次 4 個程序的 LLM 呼叫帳本 |
