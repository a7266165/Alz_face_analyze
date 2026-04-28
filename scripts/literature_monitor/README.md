# literature_monitor

本機手動觸發的文獻監測：呼叫 4 個 API（arXiv / Semantic Scholar / PubMed /
OpenAlex）按關鍵字搜四面向（embedding / asymmetry / emotion / age）的新論文，
下載 open-access PDF 到 `references/waiting_review/<topic>/<YYYYMMDD>/`，並把
結果 commit + push 到 GitHub。

## 為什麼是「本機手動」而不是 cloud routine

我們試過 Anthropic cloud routine（每天 10 個 slot 自動跑），但 cloud sandbox
的出口 IP 對 arxiv / semantic scholar / pubmed / openalex 全部回 403——academic
API 普遍對未認證的雲端流量限流。本機家用網路 egress 沒這個問題。

因此最終架構是：你在本機 review 文獻時手動跑一行指令，腳本完成搜尋+下載+
commit+push，其他裝置 `git pull` 同步。零 token 成本、簡單可預測。

## 推薦的 6 步流程（過濾優先、PDF 後抓）

直接下載所有 candidate PDF 浪費頻寬+硬碟（OA PDF 通常 50% 是雜訊）。
比較好的順序：

```bash
conda activate Alz_face_litmonitor

# (1) 抓 metadata + abstract，不下 PDF
python -m scripts.literature_monitor.run --topic all --max-per-source 50 --no-pdf

# (2)(3) abstract-based filter，雜訊搬 _archive_rejected/
python -m scripts.literature_monitor.cleanup_keywords --apply

# (4) 對 survivors 補下載 PDF
python -m scripts.literature_monitor.download_pdfs --apply

# (5)(6) 你或 Claude 讀 PDF 二次過濾（手動或 agent 輔助）
#       把內容無關的搬到 _archive_rejected/<DATE>/

# 最後：commit clean state
git add references/ && git commit -m "lit-monitor: weekly review batch" && git push
```

## 其他常用指令

```bash
# 第一次執行前：建立既有 references 索引
python -m scripts.literature_monitor.run --rebuild-index

# 跑一個特定 slot（embedding x arxiv+s2）
python -m scripts.literature_monitor.run --slot 0 --no-pdf

# 連跑 N 個 sweep（pagination cursor 自動往後翻）
python -m scripts.literature_monitor.run --topic all --max-per-source 100 --batch 20 --no-pdf

# 對特定 topic 補抓 PDF
python -m scripts.literature_monitor.download_pdfs --apply --topic emotion

# 只看不抓（dry-run）
python -m scripts.literature_monitor.run --slot 0 --dry-run --max-per-source 5

# 補回填既有 JSON 的空 abstract
python -m scripts.literature_monitor.enrich_abstracts --apply
```

## Slot 設計（10 種搜尋組合，可任意挑）

| slot | topic       | sources              | query idx | 備註     |
|------|-------------|----------------------|-----------|----------|
| 0    | embedding   | arxiv + s2           | 0         | broad    |
| 1    | asymmetry   | arxiv + s2           | 0         | broad    |
| 2    | emotion     | arxiv + s2           | 0         | broad    |
| 3    | age         | arxiv + s2           | 0         | broad    |
| 4    | embedding   | pubmed + openalex    | 1         | medical  |
| 5    | asymmetry   | pubmed + openalex    | 1         | medical  |
| 6    | emotion     | pubmed + openalex    | 1         | medical  |
| 7    | age         | pubmed + openalex    | 1         | medical  |
| 8    | all         | all                  | 2         | narrow   |
| 9    | (digest)    | —                    | —         | summary  |

實際使用上不必 10 個 slot 都跑——按你當下想找什麼 topic 挑。

## 輸出結構

```
references/waiting_review/
├── _state.json                       # 已見 paper ID（去重狀態）
├── _logs/<YYYYMMDD>.log              # 每次 run 的訊息
├── _digests/<YYYYMMDD>.md            # 每次 run 增量 digest
├── _digests/<YYYYMMDD>_summary.md    # 當日總結（slot 9 產出）
├── embedding/<YYYYMMDD>/*.pdf, *.json
├── asymmetry/<YYYYMMDD>/*.pdf, *.json
├── emotion/<YYYYMMDD>/*.pdf, *.json
└── age/<YYYYMMDD>/*.pdf, *.json
```

## 二篩流程（人工）

1. Review `_digests/<TODAY>.md` 與當日 PDF
2. **保留**：把 PDF + JSON 從 `waiting_review/<topic>/<DATE>/` 搬到
   `references/<topic>/`，並在 `references/README.md` 主目錄補一筆條目
3. **丟棄**：搬到 `references/_archive_rejected/<DATE>/`（保留紀錄，避免下輪
   再被抓）

## 既有論文索引

`references/_indexed.json` 由 `python -m scripts.literature_monitor.run
--rebuild-index` 產生，記錄 `references/<topic>/*.pdf` 的標題列表，給去重邏輯
交叉比對用。新增 / 移除 PDF 後請 rerun 此指令。

## Task Scheduler 自動化（選擇性）

如果想讓本機每天自動跑一次（不用手動觸發），可以用 Windows Task Scheduler：

1. 建立 .bat wrapper，例如 `c:/Users/4080/lit_monitor.bat`：
   ```bat
   @echo off
   call C:\Users\4080\anaconda3\Scripts\activate.bat Alz_face_litmonitor
   cd /d c:\Users\4080\Desktop\Alz_face_analyze
   python -m scripts.literature_monitor.run --topic all --max-per-source 25 --auto-push >> references\waiting_review\_logs\task_scheduler.log 2>&1
   ```
2. 在 Task Scheduler 建任務：
   - Trigger：Daily 22:00
   - Action：執行 `c:/Users/4080/lit_monitor.bat`
   - 「不論使用者登入與否」勾起來
3. 第一次手動執行 .bat 確認沒問題

不上 Task Scheduler 完全 OK——你 review 前手動跑也很乾淨。
