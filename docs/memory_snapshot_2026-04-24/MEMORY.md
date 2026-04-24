# Memory Index

- [project_details_and_discoveries.md](project_details_and_discoveries.md) — 4-arm 16-cell deep-dive 現況（2026-04-23，全 224 cells active）、cohort、statistical method per modality、pipeline 檔案、cleanup 結果。Headline：Arm B × HC ArcFace mean cosine R²=0.005(***)、Arm D × HC/NAD ArcFace full-vector Δ R²=0.008(*)、D:ACS n=20 ns、hi-lo rate-level invariance 維持。
- [project_settings_env.md](project_settings_env.md) — 12 個 conda env 對照表（分析/實驗/提取/情緒/部署）與執行慣例。experiments 用 Alz_face_test_2、不改 pyproject、新套件建 tmp env。
- [feedback_walkthrough_style.md](feedback_walkthrough_style.md) — modality 統計方法 walkthrough 時只講「特徵→test→effect size→範例」4 段，不附觀察/解讀。
- [project_external_datasets.md](project_external_datasets.md) — `external/public_face_datasets/` 已併入主 repo；2026-04-24 完成 age + emotion 抽取（MiVOLO v2 LAGENDA 無洩漏、12,245 ids），UTKFace 清理 r=0.893；4-arm + age-window classifier 都加了 `--eacs-sources` / `--arms` / `--modalities` 子集 flag；**embedding/landmark 特徵尚未抽**。
