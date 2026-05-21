# BMI 主題摘要（中文整理）

2026-05-19 由 lit-monitor 8 條 BMI query（arXiv / S2 / PubMed / OpenAlex）拉入 511 篇 → Stage 1（BMI+AD keyword）232 篇 → Stage 2（cohort n / 量化結果）146 篇 → agent triage 19 篇 ANCHOR。

標籤：`[P]` = 有 PDF 全文，`[-]` = 只 abstract / metadata。

核心 narrative（三段因果鏈）：

1. **Midlife obesity → 失智風險上升**（可修正因子，PAF 5-10%）
2. **Preclinical AD → 體重下降**（amyloid 累積驅動 BMI 下降，非低 BMI 導致 AD）
3. **Late-life obesity paradox = reverse causation + survivor bias**（APOE / sarcopenic obesity 修飾）

排列：本檔按 narrative 分群 — #1-8 midlife risk + paradox 解構、#9-13 preclinical weight loss、#14-16 PAF / population burden、#17-19 機制修飾因子。

共 **19 篇 ANCHOR**（+ 28 篇 GOOD reference 留在 waiting_review/bmi/20260519/ 未列入本檔）。

---


## ━━━━━━━━━━━━━━ Midlife risk + Late-life paradox 解構（8 篇）━━━━━━━━━━━━━━

## 1. [P] **Kim (Cho) 2022** — Association of late-life body mass index with the risk of Alzheimer disease: a 10-year nationwide population-based cohort study

> *Scientific Reports* 12:12747 | DOI: 10.1038/s41598-022-19696-2

韓國 NHIS-Senior 148,534 位 ≥65 歲老人，2002-2005 健檢，追蹤 10 年到 2015。underweight HR=1.17（1.09–1.24）、overweight HR=0.90、obese HR=0.83。進一步把 underweight 分成 mild / moderate / severe thinness，發現越瘦風險越高（HR 1.13 / 1.25 / 1.24）。

> **晚年 BMI × AD 最大單一 cohort study**，確立「晚年低體重 = AD 危險因子」+ dose-response 的關鍵 reference。

---

## 2. [-] **Karlsson 2020** — Age-dependent effects of body mass index across the adult life span on the risk of dementia

> *BMC Medicine* 18:131 | DOI: 10.1186/s12916-020-01600-2

瑞典雙胞胎登記 22,156 人（young adult → late-life BMI 軌跡）+ 美國 HRS 25,698 人，合計 47,854。Midlife（35-49 歲）每增 5 BMI units，dementia HR=1.15（1.07–1.24）；late-life（≥80 歲）HR=0.89-0.90 反而降。用 co-twin control 設計拆 genetic confounding，顯示 midlife 效果是環境/行為可修正的。

> **用 genetic design 最系統性解 paradox 的 paper**。證明 midlife BMI 效果不是 genetic confounding。

---

## 3. [-] **Park 2019** — Effect of late-life weight change on dementia incidence: a 10-year cohort study

> *BMJ Open* 9:e021739 | DOI: 10.1136/bmjopen-2018-021739

韓國 NHIS 67,219 位 60-79 歲老人，測 2002/03 與 2004/05 兩次 BMI，追蹤到 2013 看 dementia 發病。體重下降 >10% 和體重上升 >10% 都增加 dementia 風險（HR 約 1.15–1.26），呈 U 型曲線。穩定正常體重風險最低。

> **不只「瘦」本身危險，「正在變瘦或變胖」的動態過程才是 signal**。U-shaped trajectory risk at population scale。

---

## 4. [-] **Cannon 2025** — Association of BMI in Late Life, and Change from Midlife to Late Life, With Incident Dementia in the ARIC Study

> *Neurology* | DOI: 10.1212/WNL.0000000000213534

ARIC study 5,129 人，dementia-free at visit 5（2011-2013），追蹤 8 年。先看晚年 BMI：obese-stable HR=0.81（0.68–0.96）看似保護。但加入 midlife→late-life BMI 變化後，BMI 下降者 HR=2.08（1.62–2.67），原本的保護效果消失。

> **直接解開 paradox 的 paper**：late-life obesity 的「保護」是因為 preclinical dementia 導致體重下降，把下降者移出 obese 組。

---

## 5. [-] **Wu 2024** — Associations of body habitus and its changes with incident dementia in older adults

> *J Am Geriatr Soc* | DOI: 10.1111/jgs.18757

澳洲+美國 ASPREE 18,837 位 ≥65 歲健康老人（三次 anthropometric + retrospective 回溯 18 歲體重）。晚年 baseline obese HR=0.73（0.60–0.89）看似保護，但如果 18 歲和 70+ 歲都 obese（lifelong），HR 反轉為 2.27（1.22–4.24）。短期 BMI 增加 >5% 也有 HR=1.49。

> **Lifelong obesity vs late-only obesity 有完全相反的效果**。不能只看截面 BMI。

---

## 6. [P] **Zotcheva 2026** — Sex differences in BMI and waist circumference trajectories and dementia risk: HUNT4 70+

> *GeroScience* | DOI: 10.1007/s11357-025-01660-3

挪威 HUNT4 70+ cohort 9,739 人，1984-2019 共 4 次 BMI + 3 次腰圍測量。Midlife 肥胖在男女都增加 dementia 風險；late-life overweight 反而保護。女性在 dementia 診斷前 BMI 和 WC 下降幅度比男性更明顯。

> **四次重複測量 35 年**是罕見設計，直接看 midlife→late-life BMI trajectory 轉折 + 性別差異。

---

## 7. [P] **Yang 2025** — Abdominal obesity and the risk of young-onset dementia in women

> *Alzheimer's Res Therapy* | DOI: 10.1186/s13195-025-01738-2

韓國全國 964,536 名 40-60 歲女性，追蹤 8.2 年看年輕型失智（YOD）。腰圍 ≥95 cm vs <75 cm 的 YOD 風險 HR=1.55（1.34–1.79），BMI 過輕 HR=1.39、病態肥胖 HR=1.26。

> **最大 cohort 看腹部肥胖 × YOD**。腹部肥胖（內臟脂肪）比單純 BMI 更能預測中年女性失智。

---

## 8. [-] **Zhang 2024** — Sarcopenic obesity is part of obesity paradox in dementia development

> *BMC Medicine* 22:149 | DOI: 10.1186/s12916-024-03357-4

UK Biobank 60-69 歲 208,867 人。單純 obesity 在晚年看似保護（paradox），但 sarcopenic obesity（肌少+肥胖共存）dementia 風險反而高，且男女有差。高多基因風險者 sarcopenic obesity 影響更大。

> **Obesity paradox 的「保護」成分來自 lean mass preservation**，真正有害的是 sarcopenic obesity。

---


## ━━━━━━━━━━━━━━ Preclinical weight loss = AD signal（5 篇）━━━━━━━━━━━━━━

## 9. [-] **Rabin 2020** — Amyloid-beta burden predicts prospective decline in BMI in clinically normal adults

> *Neurobiology of Aging* | DOI: 10.1016/j.neurobiolaging.2020.03.002

Harvard Aging Brain Study 312 人 + ADNI 336 人，baseline amyloid PET → 追蹤 4+ 年 BMI。兩個 cohort 都顯示：baseline Aβ burden 越高 → 後續 BMI 下降越快（HABS t=−1.93 p=0.05; ADNI t=−2.54 p=0.01），調整認知表現和憂鬱後仍顯著。

> **因果反轉的 landmark paper**：不是低 BMI 導致 AD，而是 amyloid 累積先驅動體重下降。

---

## 10. [-] **Grau-Rivera 2021** — Association of weight change with CSF biomarkers and amyloid PET in preclinical AD

> *Alzheimer's Res Therapy* 13:48 | DOI: 10.1186/s13195-021-00781-z

ALFA+ 408 位認知正常中年人，4.1 年追蹤。每多 1% 體重下降，CSF p-tau 陽性 OR=1.50（1.19–1.89）、A+T+ profile OR=1.64（1.25–2.20）。認知正常但 preclinical AD 階段的人已經在掉體重。

> **直接把體重下降連結到 CSF amyloid + tau biomarker**，是 preclinical weight loss → AD pathology 最直接的生物標記證據。

---

## 11. [-] **Wang 2021** — Weight Loss and the Risk of Dementia: A Meta-analysis of Cohort Studies

> *Current Alzheimer Research* | DOI: 10.2174/1567205018666210414112723

20 篇 cohort study pooled，共 38,141 人。Weight loss 的 all-dementia RR=1.26（1.15–1.38）、AD-specific RR=1.25（1.07–1.46）。BMI 每下降 4%，風險也顯著增加。不管用哪種 weight loss 定義結果都一致。

> **體重下降 → 失智最權威的 pooled estimate**。跟 Rabin 2020 配套引用。

---

## 12. [-] **Huh 2026** — BMI levels and changes before and after dementia diagnosis and risk of all-cause mortality

> *Alzheimer's Res Therapy* | DOI: 10.1186/s13195-026-02002-x

韓國 NHIS 37,717 位新確診 dementia 患者（29,982 AD + 3,220 VaD），追蹤到 2019。診斷後 underweight 全因死亡 HR=1.57（1.46–1.69）；從 obese 降到 underweight HR=2.09（1.26–3.46）。

> **唯一大規模看確診後 BMI × 死亡的 cohort**。得了失智後體重驟降更預測死亡。

---

## 13. [-] **Baik 2025** — Twelve-year nationwide cohort study: risk factors for MCI → Alzheimer's conversion

> *Scientific Reports* | DOI: 10.1038/s41598-025-16620-2

韓國全國 MCI 患者 2009-2015 登記，追蹤到 2020。Underweight 的 MCI→DAT 轉換 HR=1.279（1.223–1.338）。70-90 歲是轉換高峰。

> **最大規模 MCI→AD 轉換研究**。Underweight 是 MCI 進展到 AD 的獨立風險因子。

---


## ━━━━━━━━━━━━━━ PAF / population burden 量化（3 篇）━━━━━━━━━━━━━━

## 14. [-] **Son 2026** — Obesity Paradox in AD: Systematic Review and Meta-Analysis of Anthropometric Measures and Age-Dependent Effects

> *Obesity Reviews* | DOI: 10.1111/obr.70078

最新 obesity paradox meta-analysis，38 篇。整體 obese OR=0.78（0.64-0.95）→ AD 風險降低，但拆年齡後 <60 歲 ES=1.65-2.45（風險增加）、>60 歲才呈保護效。Underweight ES=1.28 borderline 升高。Weight loss OR=1.31（1.08-1.58）。

> **寫 BMI × AD 論文的第一選 meta reference**。一次涵蓋所有 anthropometric measure + age stratification。

---

## 15. [P] **Stephan 2024** — Population attributable fractions of modifiable risk factors for dementia

> *Lancet Healthy Longevity* | DOI: 10.1016/S2666-7568(24)00061-8

系統回顧 + meta，48 篇。Obesity unweighted PAF=9.4%（7.3–11.7%），weighted 5.3%（3.2–7.4%）。在 14 項因子裡排第 5。

> **最完整的全球 PAF 框架**，Lancet Commission 系列更新版。定位 BMI 為「第 5 大可修正因子」。

---

## 16. [-] **Weiss 2026** — Quantifying the Dementia Burden Attributable to Excess Weight in the U.S.

> *Am J Prev Med* | DOI: 10.1016/j.amepre.2025.108231

HRS 3,734 人，用 10 年 maximum BMI window 迴避 reverse causation，追蹤 16 年。Class II/III obesity HR=1.89（1.31–2.73），PAF=22.1%（1.8–38.2%）。

> **控制 reverse causality 後肥胖 burden 大幅提高**。以前低估是因為沒控制 preclinical weight loss 干擾。

---

## 17. [-] **Li 2026** — Midlife and late-life PAF of risk factors for dementia in the U.S.

> *Alzheimer's & Dementia* | DOI: 10.1002/alz.71065

美國 DRPP 六個社區 cohort pooled 37,931 人。Midlife obesity PAF=7.7%（4.9–10.5%），是 midlife 因子裡第二大；midlife 全因子合計 22.7%。

> **量化 midlife 肥胖在美國 dementia burden 的佔比**。直接用在 paper framing。

---


## ━━━━━━━━━━━━━━ 機制修飾因子（2 篇）━━━━━━━━━━━━━━

## 18. [-] **Shinohara 2023** — APOE genotypes modify the obesity paradox in dementia

> *J Neurol Neurosurg Psychiatry* | DOI: 10.1136/jnnp-2022-331034

NACC ~20,000 人縱向追蹤 + neuropathology。APOE4 carriers：obesity 在 MCI/dementia 階段反而 protective。APOE2 carriers：obesity 加速 cognitively normal 的認知下降。Neuropathology 顯示 APOE2 carriers 的 obesity 關聯更多 microvascular pathology。

> **Obesity paradox 不是 universal，APOE 基因型決定「胖保護」vs「胖有害」**。解釋 cohort heterogeneity 必引。

---

## 19. [-] **Zeki Al Hazzouri 2021** — BMI in early adulthood and dementia in late life

> *Alzheimer's & Dementia* | DOI: 10.1002/alz.12367

CHS + Health ABC pooled 5,104 人，把 BMI 時間軸拉到 early adulthood（18-30 歲）。早期成年 BMI obese 女性 dementia OR=2.45（1.47–4.06），overweight OR=1.80（1.31–2.54），調整 midlife 和 late-life BMI 後仍然顯著。男性無此效果。

> **BMI × dementia 風險窗口延伸到 young adulthood**，且有明顯性別差異。

---


## 摘要 — BMI 主題 narrative

**Midlife risk + paradox 解構**（8 篇）：
- Kim 2022 / Karlsson 2020 / Park 2019 / Cannon 2025 / Wu 2024 / Zotcheva 2026 / Yang 2025 / Zhang 2024

**Preclinical weight loss = AD signal**（5 篇）：
- Rabin 2020（amyloid → BMI↓ landmark）/ Grau-Rivera 2021（weight loss → CSF biomarker）/ Wang 2021（weight loss meta）/ Huh 2026（post-diagnosis BMI × mortality）/ Baik 2025（MCI→AD conversion）

**PAF / population burden**（3 篇）：
- Son 2026（最新 meta）/ Stephan 2024（Lancet PAF）/ Weiss 2026 / Li 2026（美國 midlife PAF）

**機制修飾因子**（2 篇）：
- Shinohara 2023（APOE × paradox）/ Zeki Al Hazzouri 2021（early adulthood + sex）

**三段因果鏈一句話**：midlife obesity ↑ risk → preclinical AD 階段 amyloid 累積驅動體重下降 → late-life 觀測到的「胖 = 保護」是 reverse causation + survivor bias（被 APOE genotype + sarcopenic obesity 修飾）。
