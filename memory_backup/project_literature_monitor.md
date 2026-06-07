---
name: 文獻自動監測 (literature_monitor)
description: scripts/literature_monitor/ 套件，6 topic × 4 source，references/waiting_review/ 二篩
type: project
originSessionId: 3de60f8b-ee46-4b16-a6aa-7a8eebca3121
---
`scripts/literature_monitor/` 自動追蹤 6 面向（embedding/asymmetry/emotion/age/bmi/facedisease）× 4 source（arXiv/S2/PubMed/OpenAlex），PDF 下載到 `references/waiting_review/<topic>/<YYYYMMDD>/`。facedisease = 從人臉影像偵測/關聯各種疾病（與 AD-from-face 同套方法論，刻意不限縮 AD）。

## 標準流程

```bash
conda activate Alz_face_litmonitor

# (1) 抓 metadata + abstract
python -m scripts.literature_monitor.run --topic all --max-per-source 100 --batch N --no-pdf

# (2) Abstract filter
python -m scripts.literature_monitor.cleanup_keywords --apply

# (3) Agent triage（手動 spawn）

# (4) 補抓 PDF
python -m scripts.literature_monitor.download_pdfs --apply

# (5) 抽文字
conda activate Alz_face_pdftext
python -m scripts.literature_monitor.extract_text --apply --waiting-review
```

- S2 API key 存在 `Alz_face_litmonitor` env 的 `S2_API_KEY` env var
- 排程：本機手動觸發（cloud routine 因 API 403 已 disable）
- 累計 139 survivors（2026-04-29）
