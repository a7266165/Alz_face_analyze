---
name: Project details and discoveries
description: 目前狀態、核心發現。
type: project
originSessionId: 91922059-0ce1-41f4-9401-eea335107aee
---
## 現況（2026-04-21）

**三 arm 年齡控制架構**（`paper/main_v6_1_3arm.md`）：
- **Arm A** — AD vs HC naive（展示年齡主導）
- **Arm B** — AD 內部 MMSE 中位切 + 1:1 age NN 配對（橫斷面 trait-level）
- **Arm C** — baseline MMSE 中位切 + 1:1 baseline age NN（縱向 rate-level）

## Cohort

- P 患者 1,095 人 × 3,572 visits（其中 775 人 multi-visit）
- HC 候選 NAD 461 + ACS 91；strict HC = 至少一項認知評估 + (CDR=0 或 MMSE≥26)
- Arm A: AD=1,045 vs HC=278（age gap +12.8y）
- Arm B: 346 pairs from MMSE high/low within AD
- Arm C: 208 pairs from multi-visit AD with baseline MMSE split
- Kinect 雙模態資料（另一條資料線）：影片與 RGB 照片交集僅 147 人、追蹤中位 0.8 年 → 樣本 × 時程太小，未納入本研究

## 核心發現

### Arm A（naive AD vs HC）
- age_only AUC **0.864** — 任何模態皆輸或追平
- ArcFace 0.852, emotion 0.825, Dlib 0.824
- 結論：naive 高 AUC 主要是年齡代理

### Arm B（1:1 age NN matched, age p=0.81, MMSE 20.2 vs 10.1）
- **age_error** Cohen d=+0.34, paired q=0.001 ✓（重度 AD 看起來較接近實際年齡）
- **hsemotion sadness_std/range** d=-0.28 / -0.32, q<0.02 ✓（重度 AD 悲傷機率更不穩）
- **ArcFace 聯合 512-d** AUC **0.622** [0.58, 0.66]（單 dim 未過 FDR，但聯合分類可捕捉）
- 8 方法方向一致：sadness 7/8、disgust 8/8、neutral 8/8（獨立 robustness）
- 結論：**apparent age + sadness 變異度 + ArcFace embedding** 是 trait-level 存活指標

### Arm C（baseline age 1:1 NN matched, age p=0.94）
- 所有年化 drift feature **FDR q > 0.4**（null）
- best: `ann_emb_cosine_dist` d=-0.17, p=0.08（方向正確但 underpowered, power ≈62%）
- 結論：嚴格配對下 baseline 嚴重度不預測未來 drift rate

### Pool-wide longitudinal Spearman（supplementary，沿用 v6 L4）
- `emb_cosine_dist` vs ΔCDR-SB: r=+0.276, p<0.0001, n=449
- vs ΔCASI: r=-0.198; vs ΔMMSE: r=-0.127
- 跨模型：ArcFace 0.276 > Dlib 0.201 > TopoFR n.s.
- 重新定性為 "concurrent change coupling"（面部與認知同期變化耦合），而非 "severity→drift predictor"


