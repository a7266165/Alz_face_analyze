"""[LEGACY] 從 src/cohort.py 隔離出來的「下游」邏輯暫存區。

這些函式原本被塞進 cohort 模組，但職責上不屬於「人口學族群挑選」——它們
依賴下游產物（抽出的特徵、predicted_age）或屬於下游操作（配對、設計分組），
污染了 cohort 的單一職責。先搬到這裡集中隔離，待各 modality 重構時再決定
其最終歸屬或刪除。

不做 eager re-export（比照 src.common），避免把 scipy 等重 dep 帶進不需要
它的 env。Callers 直接 import 子模組：

    from src.common.legacy.predicted_age import apply_predicted_age_filter
    from src.common.legacy.feature_gate import keep_visits_with_features

（1:1 / caliper 配對已提升為 canonical 模組 src.common.matching，不再屬於本區。）
"""
