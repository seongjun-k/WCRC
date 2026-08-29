"""배포된 Flask 서버(app.py)를 리눅스에서 띄운다. app.py 는 손대지 않는다.

    python tools/serve.py                       # runs/detect/train/weights/best.pt
    python tools/serve.py path/to/other.pt

app.py 는 윈도우 경로가 박혀 있고 모델을 드래그앤드롭으로만 받는다. 그 둘만 덮어쓴다.
전처리 값은 tools/tune.json 이 있으면 그쪽이 이긴다 (wtune 이 만든다).
로봇이 붙을 주소는 이 PC 의 IP:5000 (drive.py 의 my_ip).
"""
import json
import os
import sys
import time

# 인터넷 없이(로봇 AP 만 붙은 상태로) 띄우기 위해 필요하다. ultralytics import 전에.
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2                                # noqa: E402
import numpy as np                        # noqa: E402

_TUNE = {}
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tune.json")) as _f:
        _TUNE = json.load(_f)
except (OSError, ValueError):
    pass

# ---------------------------------------------------------------- 튜닝 값
# tune.json 이 있으면 그 값, 없으면 여기 기본값. 손으로 고쳐도 된다.
CROP_RATIO = _TUNE.get("ratio", 0.75)   # 프레임 위쪽 몇 %만 쓸지 (아래는 도로)
CROP_SCALE = _TUNE.get("scale", 1)      # 1 로 둔다. imgsz 640 에서는 아무 효과가 없고
                                        # (ultralytics 가 어차피 imgsz 로 되돌린다),
                                        # imgsz 를 올린 지금은 오히려 해롭다 — 큐빅으로
                                        # 한 번 늘렸다 다시 줄이면 작은 사과가 뭉갠다.
                                        # x3 -> x1 로 바꾸니 과수원A/C 가 0 에서 1 이 됐다
                                        # (2026-08-29 대회장 사진 18장)
CLAHE_CLIP = _TUNE.get("clip", 3.0)     # 그늘 대비 보정 세기. 0 이면 안 쓴다
GAMMA = _TUNE.get("gamma", 1.0)         # 1보다 크면 밝게. 대회장은 흰 바닥·초록 매트에
                                        # 노출이 맞춰져 빨간 사과가 거의 검게 찍힌다
                                        # (사과 V=58~91 vs 화면 평균 V=167~198,
                                        # 2026-08-29 실측). CLAHE 는 국소 대비라
                                        # 이걸 못 살리고, 감마로 들어올려야 잡힌다.
NMS_IOU = _TUNE.get("iou", 0.6)         # app.py 는 iou 를 안 줘서 기본 0.7 이 된다
CONF = _TUNE.get("conf")                # None 이면 app.py 가 준 0.25 를 그대로
IMGSZ = _TUNE.get("imgsz", 1280)        # 추론 입력 해상도. 사과가 작으면 이걸 올린다.
                                        # scale 로 키워봐야 ultralytics 가 다시 imgsz 로
                                        # 줄이므로 x1 과 x3 의 결과가 완전히 같았다
                                        # (2026-08-29 대회장 사진으로 확인). 실제 레버는
                                        # scale 이 아니라 여기다.


def _brighten(img, g=GAMMA):
    """어둡게 찍힌 사과를 들어올린다. g=1 이면 아무것도 안 한다."""
    if g == 1.0:
        return img
    lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def _local_contrast(img, clip=CLAHE_CLIP):
    """밝기(L)에만 CLAHE. 색은 안 건드린다 — 빨강/초록 구분이 이 모델의 전부다."""
    if not clip:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clip, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class _CropModel:
    """모델 앞에 전처리만 끼우는 얇은 프록시. 나머지 속성은 그대로 넘긴다."""

    def __init__(self, model, ratio=CROP_RATIO, scale=CROP_SCALE):
        self._m, self._r, self._s = model, ratio, scale

    def __getattr__(self, name):
        return getattr(self._m, name)

    def __call__(self, source, **kw):
        # 크롭을 못 해도 아래 둘은 적용돼야 한다 — 전처리와 무관한 설정이다.
        kw.setdefault("iou", NMS_IOU)      # app.py 가 안 주는 값
        kw.setdefault("imgsz", IMGSZ)      # app.py 가 안 주는 값 (기본 640)
        if CONF is not None:
            kw["conf"] = CONF              # app.py 가 준 0.25 를 덮어쓴다
        im = cv2.imread(source) if isinstance(source, str) else source
        if im is None:                     # 못 읽으면 원래 동작 그대로
            return self._m(source, **kw)
        top = im[:int(im.shape[0] * self._r)]
        big = cv2.resize(top, None, fx=self._s, fy=self._s,
                         interpolation=cv2.INTER_CUBIC)
        return self._m(_local_contrast(_brighten(big)), **kw)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "wcrc 준비자료", "260814_student", "flask_server_wcrc"))

# app.py 는 import 되는 순간 모듈 최상단에서 os.makedirs(r"C:\pinky\...") 를 바로 실행한다.
# 리눅스에서는 백슬래시가 구분자가 아니라서 그 문자열 그대로 폴더 이름이 돼버린다.
# 아래에서 DIR_PT/DIR_PREDICT 를 다시 지정해도 이 시점엔 이미 늦었으니, import 하는
# 동안만 os.makedirs 를 죽여서 그 껍데기 폴더가 안 생기게 막는다.
_real_makedirs = os.makedirs
os.makedirs = lambda *a, **kw: None
try:
    import app as server                  # noqa: E402
finally:
    os.makedirs = _real_makedirs
from ultralytics import YOLO              # noqa: E402

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
PT = _args[0] if _args else os.path.join(
    ROOT, "runs", "detect", "train", "weights", "best.pt")

def make_predict_dir():
    """서버 뜰 때마다 새 폴더. 이름은 날짜시간 + 그날 몇 번째인지 (drive.py 의
    apple_shots 폴더와 같은 규칙). 파일명(타임스탬프)은 app.py 가 그대로 정한다."""
    base = os.path.join(ROOT, "runs", "server", "predict")
    os.makedirs(base, exist_ok=True)
    day = time.strftime("%Y%m%d")
    # 예전엔 사진이 이 폴더에 바로 쌓였다 — 그 파일명도 날짜로 시작해서 폴더로
    # 안 치면 개수가 같이 세인다. isdir 로 걸러야 한다.
    n = sum(1 for d in os.listdir(base)
            if d.startswith(day) and os.path.isdir(os.path.join(base, d))) + 1
    path = os.path.join(base, f"{time.strftime('%Y%m%d%H%M')}{n:02d}")
    os.makedirs(path, exist_ok=True)
    return path


# 윈도우 경로 -> 이 프로젝트 아래로
server.DIR_PT = os.path.join(ROOT, "runs", "server", "pt")
server.DIR_PREDICT = make_predict_dir()
os.makedirs(server.DIR_PT, exist_ok=True)

# 드래그앤드롭 없이 모델을 미리 물린다 (UI 에서 다른 .pt 로 바꾸는 건 그대로 된다)
assert os.path.exists(PT), f"모델이 없다: {PT}"
server.selected_pt_file = os.path.abspath(PT)
server.loaded_model = _CropModel(YOLO(server.selected_pt_file))

print(f"모델   {server.selected_pt_file}")
print(f"클래스 {server.loaded_model.names}")
print(f"결과   {server.DIR_PREDICT}")
print(f"전처리 상단 {CROP_RATIO:.0%} 크롭 x{CROP_SCALE} + CLAHE {CLAHE_CLIP} "
      f"· 감마 {GAMMA} · imgsz {IMGSZ} · NMS iou {NMS_IOU} "
      f"· conf {CONF if CONF is not None else '0.25(app.py)'}"
      + ("  (tune.json)" if _TUNE else "  (기본값)"))
print()


def selftest():
    """전처리가 실제로 걸리는지, 속성 전달이 되는지만 본다."""
    import numpy as np
    seen = {}

    class _Fake:
        names = {0: "apple"}

        def __call__(self, img, **kw):
            seen["got"] = img.shape if hasattr(img, "shape") else img
            seen["kw"] = kw
            return []

    p = _CropModel(_Fake())
    p(np.zeros((480, 640, 3), np.uint8))
    exp = (int(480 * CROP_RATIO) * CROP_SCALE, 640 * CROP_SCALE, 3)
    assert seen["got"] == exp, seen
    assert seen["kw"]["iou"] == NMS_IOU, seen         # NMS 를 조여서 넘긴다
    assert seen["kw"]["imgsz"] == IMGSZ, seen        # 작은 사과용 입력 해상도
    assert p.names == {0: "apple"}                    # 감싼 뒤에도 속성이 보인다
    p("없는파일.jpg")                                  # 못 읽으면 원본 경로 그대로 넘긴다
    assert seen["got"] == "없는파일.jpg", seen
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        server.app.run(host="0.0.0.0", port=5000, debug=False)
