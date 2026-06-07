"""Meta 端分類器:TabPFN v3 建構(指向 v3 ckpt,找不到則退回套件預設權重)。"""
from pathlib import Path

_V3_CKPT = (Path.home() / "AppData" / "Roaming" / "tabpfn"
            / "tabpfn-v3-classifier-v3_default.ckpt")


def make_tabpfn_v3(seed=42, device="auto"):
    """建立指向 v3 ckpt 的 TabPFNClassifier(找不到 ckpt 則退回套件預設權重)。"""
    from tabpfn import TabPFNClassifier
    if _V3_CKPT.exists():
        return TabPFNClassifier(model_path=str(_V3_CKPT), device=device,
                                random_state=seed, ignore_pretraining_limits=True)
    return TabPFNClassifier(device=device, random_state=seed,
                            ignore_pretraining_limits=True)
