"""배포된 Flask 서버(app.py)를 리눅스에서 띄운다. app.py 는 손대지 않는다.

app.py 는 대회 배포본이라 그대로 두는 게 맞다. 다만 그대로는 여기서 못 돈다:
  - 경로가 윈도우 하드코딩 (C:\\pinky\\pt_file)
  - 모델을 브라우저 드래그앤드롭으로만 넣을 수 있다 (tkinter 파일 다이얼로그)
그래서 import 한 뒤 그 두 가지만 덮어쓰고 실행한다.

    python tools/serve.py                       # runs/detect/train/weights/best.pt 로드
    python tools/serve.py path/to/other.pt

로봇이 붙을 주소는 이 PC 의 IP:5000 이다 (drive.py 의 my_ip).
"""
import os
import sys

import cv2                                # noqa: E402

# 추론 전에 프레임 위쪽만 잘라 2배로 키운다. app.py 는 그대로 두고 모델만 감싼다.
#
# 왜: 화면 아래 절반이 바닥이라 사과가 작게 잡힌다. 이 모델은 학습된 사과 크기
# 범위가 좁아서 그 크기에서 아예 제안을 못 만든다 (conf 를 0.05 까지 내려도 0개다.
# 신뢰도 문제가 아니다). 실측 — 나무 앞 5프레임에서 원본 0/5, 크롭 5/5 (conf 0.6~0.7).
# 지난 주행 사진 15장에서도 크롭이 원본보다 나쁜 경우가 한 번도 없었고,
# 사과 없는 프레임에서 오검출도 안 늘었다.
#
# 0.75 인 이유: 0.45 까지 조여도 결과는 같고 confidence 만 오른다. 그러면 사과가
# 잘릴 위험이 가장 적은 쪽을 고르는 게 맞다. 카메라 각도를 올리면 나무가 프레임
# 아래로 내려오므로 여유가 필요하다.
CROP_RATIO = 0.75
CROP_SCALE = 2


class _CropModel:
    """모델 앞에 전처리만 끼우는 얇은 프록시. 나머지 속성은 그대로 넘긴다.

    UI 에서 다른 .pt 를 올리면 app.py 가 loaded_model 을 생 YOLO 로 갈아끼워
    이 전처리가 빠진다. 우리는 UI 를 안 쓰므로 그대로 둔다.
    """

    def __init__(self, model, ratio=CROP_RATIO, scale=CROP_SCALE):
        self._m, self._r, self._s = model, ratio, scale

    def __getattr__(self, name):
        return getattr(self._m, name)

    def __call__(self, source, **kw):
        im = cv2.imread(source) if isinstance(source, str) else source
        if im is None:                     # 못 읽으면 원래 동작 그대로
            return self._m(source, **kw)
        top = im[:int(im.shape[0] * self._r)]
        return self._m(cv2.resize(top, None, fx=self._s, fy=self._s,
                                  interpolation=cv2.INTER_CUBIC), **kw)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "wcrc 준비자료", "260814_student", "flask_server_wcrc"))

import app as server                      # noqa: E402
from ultralytics import YOLO              # noqa: E402

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
PT = _args[0] if _args else os.path.join(
    ROOT, "runs", "detect", "train", "weights", "best.pt")

# 윈도우 경로 -> 이 프로젝트 아래로
server.DIR_PT = os.path.join(ROOT, "runs", "server", "pt")
server.DIR_PREDICT = os.path.join(ROOT, "runs", "server", "predict")
os.makedirs(server.DIR_PT, exist_ok=True)
os.makedirs(server.DIR_PREDICT, exist_ok=True)

# 드래그앤드롭 없이 모델을 미리 물린다 (UI 에서 다른 .pt 로 바꾸는 건 그대로 된다)
assert os.path.exists(PT), f"모델이 없다: {PT}"
server.selected_pt_file = os.path.abspath(PT)
server.loaded_model = _CropModel(YOLO(server.selected_pt_file))

print(f"모델   {server.selected_pt_file}")
print(f"클래스 {server.loaded_model.names}")
print(f"결과   {server.DIR_PREDICT}")
print(f"전처리 상단 {CROP_RATIO:.0%} 크롭 x{CROP_SCALE}")
print(f"추론은 app.py 가 conf=0.25 로 한다 (박스 개수만 응답)\n")


def selftest():
    """전처리가 실제로 걸리는지, 속성 전달이 되는지만 본다."""
    import numpy as np
    seen = {}

    class _Fake:
        names = {0: "apple"}

        def __call__(self, img, **kw):
            seen["got"] = img.shape if hasattr(img, "shape") else img
            return []

    p = _CropModel(_Fake())
    p(np.zeros((480, 640, 3), np.uint8))
    assert seen["got"] == (720, 1280, 3), seen        # 480*0.75*2, 640*2
    assert p.names == {0: "apple"}                    # 감싼 뒤에도 속성이 보인다
    p("없는파일.jpg")                                  # 못 읽으면 원본 경로 그대로 넘긴다
    assert seen["got"] == "없는파일.jpg", seen
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        server.app.run(host="0.0.0.0", port=5000, debug=False)
