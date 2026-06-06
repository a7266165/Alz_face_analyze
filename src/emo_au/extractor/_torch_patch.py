"""torch.load 路徑重導 contextmanager —— 規避第三方 repo 寫死的權重路徑。

OpenFace 的 RetinaFace 與 POSTER_V2 內部都 hardcode 了相對權重路徑（如
"./weights/mobilenetV1X0.25_pretrain.tar"）；載入期間暫時換掉 torch.load，比對檔名
子字串重導到實際路徑，離開即還原。
"""

from contextlib import contextmanager


@contextmanager
def patched_torch_load(redirect: dict, **load_defaults):
    """暫時 patch torch.load:path 含 redirect 某 key 子字串就換成其 value。

    load_defaults 以 setdefault 併入每次 torch.load（呼叫端已指定者不覆寫），
    供需要強制 weights_only / map_location 的第三方載入路徑使用。
    """
    import torch

    original = torch.load

    def patched(f, *args, **kwargs):
        f_str = str(f)
        for key, target in redirect.items():
            if key in f_str:
                f = target
                break
        for k, v in load_defaults.items():
            kwargs.setdefault(k, v)
        return original(f, *args, **kwargs)

    try:
        torch.load = patched
        yield
    finally:
        torch.load = original
