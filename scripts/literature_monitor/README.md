# literature_monitor

自動化文獻監測：每天 10 個 slot，按關鍵字搜尋四面向（embedding / asymmetry /
emotion / age），把候選 PDF 落地到 `references/waiting_review/<topic>/<YYYYMMDD>/`，
並可選擇 git auto-push 同步到 GitHub。

## 設計

- **薄殼 routine + 厚 Python 腳本**：所有機械工作（API query / dedup / PDF
  下載 / git push）都在 Python 裡，雲端 routine 只執行 `python -m
  scripts.literature_monitor.run --slot N --auto-push` 並讀 stdout。
- **去重**：`_state.json` 記錄已見的 paper ID（arxiv:XXX、doi:XXX、title:XXX
  fallback）。第一次跑前應先 `--rebuild-index` 掃既有 `references/<topic>/*.pdf`
  產生 `references/_indexed.json`，避免重抓已收論文。
- **PDF 取得鏈**：arXiv → Semantic Scholar `openAccessPdf` → Unpaywall →
  OpenAlex `best_oa_location`。皆無則只存 metadata `.json`。

## 使用範例

```bash
# 第一次執行前：建立既有 references 索引
python -m scripts.literature_monitor.run --rebuild-index

# Dry-run 測試（不下載、不寫 state、不 push）
python -m scripts.literature_monitor.run --slot 0 --dry-run --max-per-source 3

# 單個 slot，本地落檔但不 push
python -m scripts.literature_monitor.run --slot 0 --no-push

# 單個 slot 完整 routine 模式（包含 git push）
python -m scripts.literature_monitor.run --slot 0 --auto-push

# 手動補抓特定 topic（across all sources）
python -m scripts.literature_monitor.run --topic embedding --max-per-source 50
```

## Slot 規劃（10/day）

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

## 輸出結構

```
references/waiting_review/
├── _state.json                       # 去重狀態
├── _logs/<YYYYMMDD>.log              # 每日 log
├── _digests/<YYYYMMDD>.md            # 每 slot 增量 digest
├── _digests/<YYYYMMDD>_summary.md    # 當日 summary（slot 9 產出）
├── embedding/<YYYYMMDD>/*.pdf, *.json
├── asymmetry/<YYYYMMDD>/*.pdf, *.json
├── emotion/<YYYYMMDD>/*.pdf, *.json
└── age/<YYYYMMDD>/*.pdf, *.json
```

## 二篩流程（人工）

1. 每日 review `_digests/<TODAY>_summary.md`
2. 對保留的論文：把 PDF + JSON 從 `waiting_review/<topic>/<DATE>/` 搬到
   `references/<topic>/`，並在 `references/README.md` 主目錄補一筆條目
3. 對丟棄的論文：搬到 `references/_archive_rejected/<DATE>/`（保留紀錄，避免下輪再被抓）

## /schedule 雲端 routine 設定（待實裝階段）

routine 內容（薄殼）：
```bash
cd /path/to/Alz_face_analyze
git pull --rebase origin main
SLOT=$(( $(date -u +%H) * 10 / 24 ))   # 0..9 from UTC hour
python -m scripts.literature_monitor.run --slot $SLOT --auto-push 2>&1 | tail -n 20
```

Cron：每天 10 次（均勻分散），詳見計畫檔。
