"""[LEGACY] EACS（外部公開亞裔資料集）併入 demographics — 從 cohort 剝離。

EACS 屬於外部公開資料集；flowchart 只把它接到 Age branch，不屬於核心
P/NAD/ACS cohort。把「讀 EACS.csv 並當成 extended HC」的邏輯隔離於此，讓
``src.common.cohort`` 的 demographics 載入回歸純 P/NAD/ACS。

需要 EACS-extended demographics 的 caller（meta loader、stat grid 的
``--hc-source ACS_ext/EACS`` 等）改用本模組；或把回傳的 demo 透過
``build_cohort_ad_vs_HCgroup(..., demo=demo)`` 注入。
"""
import logging
import os

import pandas as pd

from src.config import HOSPITAL_A_CSV, DEMOGRAPHICS_DIR

VALID_HC_SOURCE_MODES = ("ACS", "ACS_ext", "EACS")

logger = logging.getLogger(__name__)


def load_combined_demographics_with_eacs(hc_source_mode="ACS_ext"):
    """Load + concat demographics across P/NAD/ACS/EACS per *hc_source_mode*.

    hc_source_mode:
      - "ACS"     : P + NAD + ACS（等同 cohort 核心，不含 EACS）
      - "ACS_ext" : P + NAD + ACS + EACS（EACS 標記為 group=ACS）
      - "EACS"    : P + NAD + EACS（以 EACS 取代內部 ACS）

    ``EACS_SOURCES`` 環境變數可逗號分隔限定 EACS 的 Source 子集。
    """
    if hc_source_mode not in VALID_HC_SOURCE_MODES:
        raise ValueError(
            f"hc_source_mode must be one of {VALID_HC_SOURCE_MODES}, "
            f"got {hc_source_mode!r}")

    frames = []
    groups_to_load = ["P", "NAD"]
    if hc_source_mode != "EACS":
        groups_to_load.append("ACS")
    hospital_A = pd.read_csv(HOSPITAL_A_CSV)
    for grp in groups_to_load:
        df = hospital_A[hospital_A["Group"] == grp].copy()
        # hospital_A 為 split schema；組回完整特徵 ID（"P1-2"）供 match / 特徵對應。
        df["ID"] = (df["Group"] + df["ID"].astype(str)
                    + "-" + df["Photo_Session"].astype(str))
        df["group"] = grp
        df["Source"] = "internal"
        frames.append(df)
    if hc_source_mode in ("ACS_ext", "EACS"):
        df_e = pd.read_csv(DEMOGRAPHICS_DIR / "EACS.csv")
        eacs_sources_env = os.environ.get("EACS_SOURCES", "").strip()
        if eacs_sources_env:
            wanted = {s.strip()
                      for s in eacs_sources_env.split(",") if s.strip()}
            if "Source" in df_e.columns:
                df_e = df_e[df_e["Source"].isin(wanted)].copy()
                logger.info(
                    f"filtered EACS to sources {wanted}: {len(df_e)} rows")
        df_e["group"] = "ACS"
        frames.append(df_e)
    demo = pd.concat(frames, ignore_index=True)
    if "Source" not in demo.columns:
        demo["Source"] = "internal"
    demo["Source"] = demo["Source"].fillna("internal")
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["Global_CDR"] = pd.to_numeric(
        demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    demo["base_id"] = demo["ID"].str.extract(r"^(.+)-\d+$")
    demo["visit"] = demo["ID"].str.extract(r"-(\d+)$").astype(float)
    return demo


def cohort_list_with_eacs(hc_source_mode, p_visit, p_score, hc_visit,
                          hc_score):
    """EACS-extended 版 ``cohort_list``（HC 含 EACS）。

    回傳 cohort 的 6 欄 + 額外的 ``group`` 欄——因為 EACS ID
    （EACS_UTKFace_00191-1）無法用 ``^([A-Za-z]+)\\d`` 拆出 group，必須沿用
    loader 指定的 group（EACS→ACS）。下游 / match 對這份 roster 應直接用
    ``group`` 欄，不要再從 ID 重算。
    """
    from src.common.cohort import (
        _OUTPUT_COLS,
        hc_filter,
        p_filter,
        visit_selection,
    )

    demo = load_combined_demographics_with_eacs(hc_source_mode)
    demo = demo[demo["Age"].notna()].copy()
    p = visit_selection(p_filter(demo[demo["group"] == "P"], p_score), p_visit)
    hc = visit_selection(
        hc_filter(demo[demo["group"] != "P"], hc_score), hc_visit)
    return pd.concat([p, hc], ignore_index=True)[
        _OUTPUT_COLS + ["group"]].reset_index(drop=True)
