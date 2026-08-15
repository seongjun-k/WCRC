"""핑키 로봇에서 데이터셋용 사진 모으기. 로봇의 Jupyter/터미널에서 실행한다.

    python capture.py [저장폴더] [장수] [간격초]
    python capture.py raw/images 30 1.0      # 기본값

촬영 중 PC 브라우저에서 http://<로봇IP>:8080 을 열면 실시간 프리뷰가 보인다.
(PC에서는 pinky-cam 이 브라우저를 자동으로 띄운다)

찍는 동안 로봇을 조금씩 옮기고 각도·거리를 바꾼다.
사과가 없는 배경 프레임(잎, 마커, 다른 로봇)도 섞어 찍을 것 — 오탐 억제용.
"""
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from pinkylib import Camera

out = sys.argv[1] if len(sys.argv) > 1 else "raw/images"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
PORT = 8080

os.makedirs(out, exist_ok=True)

latest = None  # 프리뷰로 내보낼 최신 JPEG


class Preview(BaseHTTPRequestHandler):
    """MJPEG 스트림. 브라우저가 <img src> 하나로 바로 받아준다."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                if latest:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest + b"\r\n"
                    )
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 브라우저 탭을 닫은 것뿐 — 촬영은 계속한다

    def log_message(self, *args):
        pass


threading.Thread(
    target=lambda: HTTPServer(("", PORT), Preview).serve_forever(), daemon=True
).start()
print(f"프리뷰: http://<로봇IP>:{PORT}")

cam = Camera()
# set_calibration() 은 아루코 pose 계산용 행렬만 읽는다. get_frame() 결과에는 영향이 없어
# (캡처 + 180도 회전이 전부) 데이터 수집에는 부르지 않는다 — npz 파일도 필요 없다.
cam.start()

saved = 0
next_shot = time.time()
try:
    # 프레임은 계속 읽어 프리뷰를 갱신하고, 저장만 interval 간격으로 한다.
    # (촬영 간격에 맞춰 읽으면 프리뷰가 1초에 한 번씩 끊겨 위치를 잡기 어렵다)
    while saved < count:
        frame = cam.get_frame()
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            latest = buf.tobytes()

        if time.time() >= next_shot:
            path = os.path.join(out, f"{time.strftime('%Y%m%d_%H%M%S')}_{saved:03d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            next_shot = time.time() + interval
            print(f"[{saved}/{count}] {path}", flush=True)
finally:
    cam.close()

print(f"\n완료. {out} 를 PC로 옮겨서 dataset.py label 실행")
