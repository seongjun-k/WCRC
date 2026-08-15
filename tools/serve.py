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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "wcrc 준비자료", "260814_student", "flask_server_wcrc"))

import app as server                      # noqa: E402
from ultralytics import YOLO              # noqa: E402

PT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "runs", "detect", "train", "weights", "best.pt")

# 윈도우 경로 -> 이 프로젝트 아래로
server.DIR_PT = os.path.join(ROOT, "runs", "server", "pt")
server.DIR_PREDICT = os.path.join(ROOT, "runs", "server", "predict")
os.makedirs(server.DIR_PT, exist_ok=True)
os.makedirs(server.DIR_PREDICT, exist_ok=True)

# 드래그앤드롭 없이 모델을 미리 물린다 (UI 에서 다른 .pt 로 바꾸는 건 그대로 된다)
assert os.path.exists(PT), f"모델이 없다: {PT}"
server.selected_pt_file = os.path.abspath(PT)
server.loaded_model = YOLO(server.selected_pt_file)

print(f"모델   {server.selected_pt_file}")
print(f"클래스 {server.loaded_model.names}")
print(f"결과   {server.DIR_PREDICT}")
print(f"추론은 app.py 가 conf=0.25 로 한다 (박스 개수만 응답)\n")

if __name__ == "__main__":
    server.app.run(host="0.0.0.0", port=5000, debug=False)
