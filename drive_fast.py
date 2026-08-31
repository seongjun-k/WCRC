# !/usr/bin/env python3
"""2026 WCRC 주행 코드 (Pinky pro).

대회장에서 (노트북 터미널):
    wrun            # 주행 — 서버 기동 · 코드 업로드 · 점검까지 한 번에
    wtune 1 2 1     # 직전 주행 사진으로 사과 인식 재튜닝 (과수원별 실제 개수)

로봇에서 (ssh pinky@192.168.4.1):
    python3 drive.py check        # 출발 전 점검 (모터 안 돎)
    python3 drive.py run          # 주행 (준비 끝나고 엔터에서 출발)e
    python3 drive.py run now      # 엔터 없이 바로
    python3 drive.py pose         # 보이는 마커의 x, z — HOP / END_EXTRA_CM 보정용

단계별 테스트:  motors -> track 1 -> actions 0 -> run
측정:          forward-cal (연속) / road-cal (도로추종·실제값) / turn-cal / teleop / road
PC 검증:       python tools/sim_drive.py  /  python tools/road.py
"""
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# 하드웨어 모듈은 setup() 안에서 늦게 import 한다.
pinky_cam = pinky_motor = pinky_lcd = pinky_buzzer = pinky_imu = None
SERVER_URL = None

# ================================================================ 여기만 고친다
# 대회장에서 실제로 만지는 값들. 그 아래 "그 외 상수" 는 건드릴 일이 거의 없다.

# Flask 서버 PC 의 IP. go.sh 가 자기 주소를 WCRC_SERVER_IP 로 넘겨준다 (노트북이
# 바뀌어도 코드를 안 고쳐도 된다). 손으로 돌릴 땐 아래 기본값을 쓴다.
my_ip = os.environ.get("WCRC_SERVER_IP", "192.168.4.12")

MOVE_FORWARD_PER_ONE = 19.69   # [FAST] 17.72 x 100/90 선형 추정. ★★ 개루프라 이 값이 틀리면
                               # 충돌한다. 매트에서 `forward-cal` 또는 `road-cal` 로 반드시
                               # 재측정 후 쓸 것. 원본은 17.72 (속도 90 기준).
                               # 냉간 첫 이동은 13.2 로 26% 느리다

# 마커에 정렬한 뒤 눈 감고 더 가는 거리(cm) = 교차로 정지 위치. 마커별로 하나씩.
# 값의 정의: 마커를 고정한 순간의 실제 z. 주행 로그의 z 를 보고 고친다.
# 마커5(하역장): 36(짧음) -> 63(부딪힘/이탈) -> 50(여전히 10~20cm 김, 2026-08-28
# 실주행 피드백) -> 그 중간값인 15cm 를 빼서 35 로. 원래 값(36)과 거의 같아졌다 —
# 이번에도 눈대중 보정이라 다음 주행에서 다시 확인할 것.
HOP = {1: 43, 2: 43, 4: 35, 5: 35, 10: 17}

# END 위치 = 마커10 의 z + 이 값. 음수면 마커 앞에서 선다.
# HOP[10] 을 바꿔도 END 는 안 밀린다 (remain = z - hop + extra 로 상쇄).
# -11 -> -6 으로 5cm 더 가게 (2026-08-28 실주행 피드백).
END_EXTRA_CM = -6              # END 자세에서 pose 로 재니 z=14 였다 (2026-08-27)

PEEK_TIME = 0.15               # 과수원에 코 들이미는 전진/후진 시간(초). x17.72 = 2.7cm
APPLE_CHECK_COUNT = 3          # 사과를 몇 장 찍어 중앙값을 쓸지. 반드시 홀수
MOTOR_SPEED = 100              # [FAST] 최대 100. ★ 도로 조향이 포화(안쪽 바퀴만 줄여 약하게 보정).
                               # ★ 개루프! 아래 MOVE_FORWARD_PER_ONE 을 반드시 같이 올려야
                               #   거리가 맞는다. 안 맞으면 오버슈트=충돌.
TURN_MAX_SPEED = 100           # [FAST] 60->100. IMU 폐루프라 안 부딪힌다. 회전이 병목이라
                               # 여기가 최대 이득. 오차 크면 보정 회전 1회가 붙을 수 있으니
                               # ★ 테스트 랩에서 회전 오버슈트 확인. 심하면 TURN_COAST_PER_SPEED 를
                               #   이 속도(100)에서 turn-cal 로 다시 재기 (0.248 은 60 기준)
                               # 45 로 낮췄다가 (2026-08-28) 오히려 굼뜨고, 계단식 가속은
                               # move() 호출 사이가 끊겨 보여서 둘 다 원래 값·방식으로 되돌렸다
SEARCH_COUNT = 10              # 마커 탐색 스텝 수. 10 x 6도 = 60도

# ================================================================ 그 외 상수

LEFT = 0
RIGHT = 1
FORWARD = 2

# after_track_list 에서 사용하는 동작 코드
GO_STRAIGHT = 100         # 전진
MOVE_RIGHT = 101          # 우회전
MOVE_LEFT = 102           # 좌회전
GO_BACKWARD = 103         # 후진
APPLE_COUNT_ACTION = 104  # 사과 개수 세기 (서버로 이미지 전송, 개수 응답)
APPLE_DISPLAY = 105       # LCD 에 사과 개수 표시
CROSS_WALK_WAIT = 106     # 대기 (횡단보도). 초 단위 인자를 받는다
GO_TO_MARKER = 107        # 마커를 잡았을 때 잰 z 만큼 전진 (옵션 = 더 갈 cm)
CHECK_NEXT = 108          # 다음 목표 마커가 보이는지 그 자리에서 확인만
DEFAULT = 999             # 시간 옵션이 필요 없는 동작

SEARCH_MOTOR_SPEED = 65

# 제자리 회전은 시간이 아니라 각도로 지정하고 IMU(BNO055)로 닫는다.
TURN_COAST_PER_SPEED = 0.248   # 모터를 끈 뒤 관성으로 더 도는 양. 실측
                               # 2026-08-29 경기장 매트에서 재측정: 학교바닥 0.228 은
                               # 여기선 매회 +1.4도 초과(90x4 누적 +5.6). 매트가 덜 미끄러워
                               # coast 가 커서 0.248 로 올림 (초과 ~0 수렴).
TURN_COAST_RATIO = 0.30        # coast 가 목표각의 이 비율을 넘지 않게 속도를 정한다
TURN_MIN_SPEED = 10
TURN_SETTLE_BASE = 0.10        # 회전 뒤 대기 = BASE + PER_SPEED * speed
TURN_SETTLE_PER_SPEED = 0.004
TURN_TOL_DEG = 1.5             # 이 안이면 도달. 넘으면 보정 회전 1회
TURN_MAX_SEC = 4.0             # IMU 가 이상해도 여기서 끊는다
TURN_DEG_PER_SEC = 212         # IMU 를 못 쓸 때의 시간 기반 폴백

SEARCH_STEP_DEG = 6            # 탐색 한 스텝의 회전 각도
SEARCH_SCAN_SPEED = 25         # 낮을수록 덜 지나친다
SEARCH_RETURN_DEG = SEARCH_STEP_DEG * SEARCH_COUNT   # 실패하면 훑은 만큼 되돌린다
FIND_ADVANCE_CM = 8            # 회전만으로 못 찾으면 조금 전진해서 다시 본다
FIND_ADVANCE_TRIES = 3
MATCH_MAX_DEG = 30             # 정렬 회전 1회 상한
MATCH_MIN_DEG = 2              # 이보다 작으면 coast 에 묻힌다
MATCH_FORWARD_TIME = 0.4

SLEEP_TIME_AFTER_MOVE = 0.03   # 액션마다 무조건 자던 값. 현재 액션 24개 x 0.08 = 1.92초를
                               # 그냥 서 있었다. 회전은 자체 정지대기(0.34초)가 따로 있고
                               # 전진도 끝나면 모터가 꺼진 상태라 이중 대기였다 (2026-08-29)
MOTOR_BIG_STEP_FORWARD = 1
STRAIGHT_TO_MAIN_ROAD_TIME = 3

# 한 번에 전진할 최대 시간(초). 잘라서 가며 매번 다시 재야 MOVE_FORWARD_PER_ONE 이
# 좀 틀려도 수렴한다. 한 방에 가면 그 사이 마커를 못 본다.
APPROACH_MAX_STEP = 0.5

Z_ACCEPT_MIN = 8               # 이 밖의 z 는 오인식으로 버린다.
Z_ACCEPT_MAX = 85              # 36mm 마커는 실측상 37~75cm 에서만 제대로 읽힌다
MARKER_TIMEOUT = 35            # 마커 하나에 매달릴 최대 초. 규정상 60초면 기회 종료
MARKER_SIZE_M = 0.036          # 실측 36mm. 틀리면 모든 z 가 같은 배율로 어긋난다
MARKER_SIDE_OFFSET = 12        # 마커를 도로 중심선에서 옆으로 떼는 거리(cm)
CALIBRATION_PATH = "/home/pinky/CH/camera_calibration.npz"

APPLE_SHOT_GAP = 0.02          # 사진 사이 간격(초)
APPLE_JOIN_TIMEOUT = 3.0          # 6.0 -> 3.0. 마커1·2 사진은 이미 끝나 있어 join 이 0초라,
                                 # 실제로 기다리는 건 마커4 것 하나뿐이다 (그것도 하차장까지
                                 # 오는 4~5초를 이미 벌었다). 1.5 까지 조였다가 되돌렸다 —
                                 # 개수 하나 놓치면 규정 3번 패널티 30초라 기다리는 게 싸다.
                                 # 3.0 이면 최악도 9초(3장x3초), 6.0 일 때의 18초보다 낫다
APPLE_SHOT_DIR = "/home/pinky/apple_shots"   # 주행마다 그 아래에 폴더 하나
run_dir = APPLE_SHOT_DIR                     # 실제 저장 위치. setup() 이 정한다


def _log(*a):
    """진단 메시지를 화면 대신 파일에만 남긴다 (2026-08-29).

    "마커 안 보임", "도로 안 보임" 같은 실패 로그가 터미널에 뜨면 옆 팀이 보고
    시비를 걸 수 있다. 값 자체는 다음 주행 보정에 필요하니 버리지는 않는다.
    go.sh 가 주행 뒤 사진 폴더를 통째로 가져오므로 이 파일도 같이 딸려온다.
    """
    try:
        with open(f"{run_dir}/run.log", "a", encoding="utf-8") as f:
            f.write(" ".join(str(x) for x in a) + "\n")
    except OSError:
        pass

# 규정 3번: 잘 익은 사과 = 빨간 사과만 카운팅
total_apple_count = 0

# ---------------------------------------------------------------- 주행 순서

# START -> 교차로1(1) -> 2 -> 3(4) -> 4(5) -> 횡단보도(10) -> END
# 과수원 마커(3,6,9)와 하차장 마커(0)는 목표에서 뺐다. 교차로 마커에 정렬한 그 자리에서
# 90도 돌아 코만 들이밀고 후진으로 빠진다(after_track_list). 180도 회전이 없어지고
# 후진이라 복귀 후 헤딩도 안 틀어진다. 찾을 마커도 9개 -> 5개.
#
# pose = [x, z, 탐색_시작_방향], 단위 cm. 마커를 못 봤을 때만 쓰는 fallback 이라
#   주행 로그의 `다음 마커 N 확인 (z=..)` 실측으로 갱신할 것.
#   탐색 방향은 x 의 부호와 같아야 한다 (x<0 이면 LEFT, x>0 이면 RIGHT).
#   x 를 0 으로 두면 마커를 화면 가운데 놓으려고 로봇이 도로에서 20도쯤 틀어진다.
#
# 36mm 마커는 37~75cm 에서만 읽힌다 (가까우면 초점이 안 맞고 멀면 픽셀이 모자란다).
# 마커는 교차로 옆 12~20cm 에만 놓을 수 있어 "마커 앞 N cm 정지" 가 성립하지 않는다.
# 그래서 2단계다: 보이는 동안 접근해 위치를 확정(z) -> 남은 거리는 눈 감고 간다(HOP).
target_list = [
    {"id": 1,  "pose": [ +4, 20, RIGHT], "cm": 0, "hop": HOP[1],
     "peek": 0, "confirm": True},                                # 교차로1 (과수원A)
    {"id": 2,  "pose": [+16, 34, RIGHT], "cm": 0, "hop": HOP[2],
     "peek": 0, "confirm": True},                                # 교차로2 (과수원B)
    {"id": 4,  "pose": [-11, 40, LEFT ], "cm": 0, "hop": HOP[4],
     "peek": 0, "confirm": True},                                # 교차로3 (과수원C)
    {"id": 5,  "pose": [ -1, 20, LEFT ], "cm": 0, "hop": HOP[5],
     "peek": 0, "confirm": True},                                # 교차로4 (하차장)
    {"id": 10, "pose": [-18, 64, LEFT ], "cm": 0, "hop": HOP[10],
     "peek": 0, "confirm": True},                                # 횡단보도 정지선
]


# 마커 도착 후 수행할 동작. target_list 와 개수·순서가 같아야 한다 (지금 5개).
#
# 과수원·하차장 공통 패턴 — 가지 안으로 들어가지 않는다:
#   90도 회전 → 살짝 전진(PEEK_TIME) → 임무 → 같은 시간 후진 → 90도 반대 회전 → 주행 재개
# 회전 옵션의 단위는 **도(degree)** 다. IMU 로 닫아서 실제로 그 각도만큼 돈다.
#
# 후진으로 빠지는 게 핵심이다. 180도 돌아 나오면 회전 오차가 그대로 진행 방향 오차가
# 되는데, 후진은 방향을 건드리지 않아 원래 헤딩 그대로 다음 교차로를 향한다.
#
# 회전 방향은 맵 기준이다. 과수원 A·C 는 메인도로 남쪽, B 와 하차장은 북쪽이라 반대로 돈다.
# ★ 코스에서 반드시 눈으로 확인할 것. 방향이 틀리면 반대편 허공을 보고 사과를 센다.
after_track_list = [
    # 교차로1 → 과수원A (남쪽): 오른쪽으로 돌아본다
    {"id": 1,  "actions": [(MOVE_RIGHT, 90),
                           (GO_STRAIGHT, PEEK_TIME),
                           (APPLE_COUNT_ACTION, DEFAULT),
                           (GO_BACKWARD, PEEK_TIME),
                           (MOVE_LEFT, 90),
                           (CHECK_NEXT, DEFAULT)]},

    # 교차로2 → 과수원B (북쪽): 왼쪽
    {"id": 2,  "actions": [(MOVE_LEFT, 90),
                           (GO_STRAIGHT, PEEK_TIME),
                           (APPLE_COUNT_ACTION, DEFAULT),
                           (GO_BACKWARD, PEEK_TIME),
                           (MOVE_RIGHT, 90)]},

    # 교차로3 → 과수원C (남쪽): 오른쪽
    {"id": 4,  "actions": [(MOVE_RIGHT, 90),
                           (GO_STRAIGHT, PEEK_TIME),
                           (APPLE_COUNT_ACTION, DEFAULT),
                           (GO_BACKWARD, PEEK_TIME),
                           (MOVE_LEFT, 90)]},

    # 교차로4 → 하차장 (북쪽): 왼쪽.
    {"id": 5,  "actions": [(MOVE_LEFT, 90),
                           (GO_STRAIGHT, PEEK_TIME),
                           (APPLE_DISPLAY, DEFAULT),
                           (GO_BACKWARD, PEEK_TIME),
                           # 80+10 으로 쪼개 중간에서 마커10 을 보던 걸 합쳤다.
                           # 쪼개면 10도짜리가 speed 13 으로 느리게 돌아 0.3~0.5초를 먹는다.
                           # 다 돌고 봐도 마커10 은 15도 왼쪽이라 화면 안이다 (교차로1 은 25도).
                           (MOVE_RIGHT, 90),
                           (CHECK_NEXT, DEFAULT)]},

    # 횡단보도 정지선에 섰다.
    {"id": 10, "actions": [(CROSS_WALK_WAIT, 3.0),
                           (GO_TO_MARKER, END_EXTRA_CM)]},
]


# ================================================================ 서버 통신

def get_server_url(ip):
    return f"http://{ip}:5000/predict"

def send_image_and_get_count(image_input, retries=2, retry_wait=0.5):  # 이미지를 PC(Flask)로 전송하고 감지된 사물의 총 개수(count)를 받아옵니다.
    # wifi 가 순간 끊기면(수 초) 통신 예외 한 번으로 그 사진은 그냥 버려져 과수원이
    # 통째로 0개 처리됐다 (2026-08-28 실주행에서 확인). 통신 예외일 때만 짧게 재시도한다
    # — 서버가 정상 응답한 400/기타 에러는 재시도해도 똑같으니 그대로 포기한다.
    for attempt in range(retries):
        try:
            # 1) 파일 경로(문자열)인 경우
            if isinstance(image_input, str):
                with open(image_input, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(SERVER_URL, files=files, timeout=3)

            # 2) Picamera2 / OpenCV frame (Numpy Array)인 경우
            else:
                # 메모리 상에서 즉시 JPEG 바이너리로 인코딩 (파일 저장 없이 고속 처리)
                success, img_encoded = cv2.imencode('.jpg', image_input)
                if not success:
                    _log(" image encoding failed")
                    return None

                files = {'image': ('robot_frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
                response = requests.post(SERVER_URL, files=files, timeout=3)

            # 응답 처리
            if response.status_code == 200:
                res_data = response.json()

                # [디버그 출력] 서버가 실제로 보내온 데이터 전체를 확인합니다.
                print("Server respose raw data:", res_data)

                # 서버에서 'detected_count' 키값을 가져옴
                if 'detected_count' in res_data:
                    count = res_data['detected_count']
                else:
                    _log("warning:'detected_count' key is missing in Server response")
                    count = 0

                print(f"detected count: {count} (file name: {res_data.get('saved_filename')})")
                return count

            elif response.status_code == 400:
                err_msg = response.json().get('message', '요청 에러')
                print(f" Error : {err_msg}")
                return None

            else:
                _log(f" Server status error (code {response.status_code}):", response.text)
                return None

        except Exception as e:
            _log(f"Communication exception (시도 {attempt + 1}/{retries}):", e)
            if attempt + 1 < retries:
                time.sleep(retry_wait)
    return None

# ================================================================ LCD / 사과

# LCD 폰트. NanumGothic.ttf 는 로봇에 없다 (없으면 기본 비트맵 폰트라 글씨가 아주 작다).
LCD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _fit_font(draw, text, box_w, box_h):
    """주어진 칸을 꽉 채우는 폰트 크기를 찾는다 (없으면 기본 폰트)."""
    if not os.path.exists(LCD_FONT):
        return ImageFont.load_default()
    lo, hi, best = 8, 400, None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(LCD_FONT, mid)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        if r - l <= box_w and b - t <= box_h:
            best, lo = f, mid + 1
        else:
            hi = mid - 1
    return best or ImageFont.truetype(LCD_FONT, 8)


def display_apple_count(apple_count):
    """규정 6번: 센 사과 개수를 LCD 에 표시한다."""
    W, H = 320, 240
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    label = "APPLES"
    num = str(apple_count)

    fl = _fit_font(draw, label, int(W * 0.9), int(H * 0.16))
    l, t, r, b = draw.textbbox((0, 0), label, font=fl)
    draw.text(((W - (r - l)) // 2 - l, 6 - t), label, fill=(0, 220, 0), font=fl)

    top = int(H * 0.20)
    fn = _fit_font(draw, num, int(W * 0.9), int(H * 0.74))
    l, t, r, b = draw.textbbox((0, 0), num, font=fn)
    draw.text(((W - (r - l)) // 2 - l, top + (H - top - (b - t)) // 2 - t),
              num, fill=(255, 255, 255), font=fn)

    pinky_lcd.img_show(img)


# 주행할 때마다 마커별 "목표 z vs 실제로 멈춘 z" 를 모은다.
_cal = []
# 주행 시간이 어디로 가는지 재는 계측기. 함수를 감싸기만 하므로 거동은 안 바뀐다.
_timing = {}


def _timed(label):
    def deco(fn):
        def wrap(*a, **kw):
            t = time.time()
            try:
                return fn(*a, **kw)
            finally:
                _timing[label] = _timing.get(label, 0.0) + time.time() - t
        wrap.__name__ = fn.__name__
        wrap.__doc__ = fn.__doc__
        return wrap
    return deco


_last_arrival_z = None
_last_arrival_id = None    # 그 z 가 어느 마커의 것인지. 덮어쓰기 사고를 막는다

_apple_jobs = []


@_timed("사진촬영")
def start_apple_count():
    """사진만 찍어두고 서버 전송은 백그라운드로 넘긴다."""
    frames = []
    for i in range(APPLE_CHECK_COUNT):
        frames.append(pinky_cam.get_frame())
        if i < APPLE_CHECK_COUNT - 1:
            time.sleep(APPLE_SHOT_GAP)   # 완전히 같은 프레임 3장은 최댓값의 의미가 없다
    # 주행이 끝난 뒤 "로봇이 뭘 보고 셌나" 를 확인할 수 있어야 한다.
    n = len(_apple_jobs) + 1
    for i, f in enumerate(frames):
        try:
            cv2.imwrite(f"{run_dir}/apple_{n}_{i}.jpg", f)
        except Exception as e:
            _log("  사진 저장 실패:", e)
    print(f"  사진 {len(frames)}장 저장 -> {run_dir}/apple_{n}_*.jpg")
    box = {}

    def work():
        # 중앙값을 쓰면 3장 중 2장이 각도·가림으로 덜 잡았을 때 진짜 개수(최댓값)가
        # 묻힌다 — 실주행에서 4개가 2개로 나온 원인 (2026-08-28). 최댓값으로 되돌린다.
        cs = [c for c in (send_image_and_get_count(f) for f in frames) if c is not None]
        box["n"] = max(cs) if cs else 0

    th = threading.Thread(target=work, daemon=True)
    th.start()
    _apple_jobs.append((th, box))


@_timed("사과집계대기")
def collect_apple_counts():
    """백그라운드로 보낸 사과 개수를 전부 거둬 합계를 돌려준다."""
    total = 0
    for th, box in _apple_jobs:
        th.join(timeout=APPLE_JOIN_TIMEOUT)
        if th.is_alive():
            _log("  [경고] 사과 개수 서버 응답 없음 — 그 과수원은 0개로 친다")
        total += box.get("n", 0)
    _apple_jobs.clear()
    return total

# ================================================================ 모터

def move_forward(duration_time, motor_speed=MOTOR_SPEED):
    pinky_motor.move(motor_speed, motor_speed)
    time.sleep(duration_time)
    pinky_motor.move(0, 0)

def move_backward(duration_time, motor_speed=MOTOR_SPEED):
    pinky_motor.move(-motor_speed, -motor_speed)
    time.sleep(duration_time)
    pinky_motor.move(0, 0)

def move_right(duration_time, motor_speed=MOTOR_SPEED):
    pinky_motor.move(motor_speed, -motor_speed)
    time.sleep(duration_time)
    pinky_motor.move(0, 0)

def move_left(duration_time, motor_speed=MOTOR_SPEED):
    pinky_motor.move(-motor_speed, motor_speed)
    time.sleep(duration_time)
    pinky_motor.move(0, 0)


def read_yaw():
    """IMU 의 yaw(도). 못 읽으면 None. 6-DOF 모드라 자이로 적분값이다."""
    if pinky_imu is None:
        return None
    for _ in range(3):
        d = pinky_imu.read_imu_data()
        if d:
            return d["euler"][2]
    return None


def turn_speed_for(deg):
    """이 각도를 돌기에 알맞은 속도. coast 가 목표각을 잡아먹지 않게 고른다."""
    ideal = abs(deg) * TURN_COAST_RATIO / TURN_COAST_PER_SPEED
    return int(max(TURN_MIN_SPEED, min(TURN_MAX_SPEED, ideal)))


def _turn_once(deg, speed):
    """한 번 돌고 실제로 돈 각도를 돌려준다 (coast 포함)."""
    y0 = read_yaw()
    if y0 is None:
        t = abs(deg) / (TURN_DEG_PER_SEC * speed / MOTOR_SPEED)
        (move_right if deg > 0 else move_left)(t, speed)
        return deg

    turned, prev = 0.0, y0
    target = max(0.0, abs(deg) - TURN_COAST_PER_SPEED * speed)   # coast 만큼 미리 멈춘다
    deadline = time.time() + TURN_MAX_SEC
    sign = 1 if deg > 0 else -1
    # 계단식 가속을 넣었더니 move() 를 여러 번 나눠 부르는 사이가 실제로 끊겨 보였다
    # (2026-08-28). 그냥 한 번에 목표 속도로 붙인다 — 원래 방식.
    pinky_motor.move(speed * sign, -speed * sign)
    while abs(turned) < target and time.time() < deadline:
        y = read_yaw()
        if y is not None:
            turned += (y - prev + 180) % 360 - 180   # 360도 넘김 처리
            prev = y
        time.sleep(0.01)
    pinky_motor.move(0, 0)

    time.sleep(TURN_SETTLE_BASE + TURN_SETTLE_PER_SPEED * speed)  # coast 끝날 때까지
    y = read_yaw()
    if y is not None:
        turned += (y - prev + 180) % 360 - 180
    return turned


@_timed("회전")
def turn_deg(deg):
    """제자리에서 deg 도 돈다 (양수 = 우회전). 실제로 돈 각도를 돌려준다."""
    if abs(deg) < 0.5:
        return 0.0

    turned = _turn_once(deg, turn_speed_for(deg))
    err = deg - turned
    if read_yaw() is not None and abs(err) > TURN_TOL_DEG:
        turned += _turn_once(err, turn_speed_for(err))

    print(f"  회전 목표 {deg:+.0f}도 -> 실제 {turned:+.1f}도")
    return turned


def turn_left_deg(deg):
    return turn_deg(-abs(deg))


def turn_right_deg(deg):
    return turn_deg(abs(deg))


def go_straight_to_main_road(duration_time=STRAIGHT_TO_MAIN_ROAD_TIME):
    motor_speed = MOTOR_SPEED
    move_forward(duration_time)
    time.sleep(duration_time)
    pinky_motor.move(0, 0)

# ================================================================ 도로 유지


# ▼ 대회장 조명에서 재조정할 값 2개 (tools/road.py --tune 이 추천해준다) 차선(도로 마스크) 추종을 쓸지. False 면 그냥
# 직진한다.
ROAD_FOLLOW = True

# 마커까지 외워둔 거리 중 몇 %를 카메라 없이 먼저 달릴지. 나머지는 마커로 폐루프. 1.0 에 가까울수록 빠르지만 마커를 지나칠 위험이 커진다.
APPROACH_BLIND_RATIO = 0.85

ROAD_AUTO = True    # 프레임마다 Otsu 로 임계값을 잡는다. 이상하면 False
ROAD_S_MAX = 70          # 이보다 채도가 높으면 도로가 아니다 (잔디·테두리·집기)
ROAD_V_MIN = 150         # 이보다 어두우면 도로가 아니다 (그림자)

ROAD_NEAR_BAND = (0.80, 1.00)   # 바로 앞  — 좌우 치우침 계산용
ROAD_FAR_BAND = (0.60, 0.78)    # 조금 먼 앞 — 도로가 휘는 방향 계산용
ROAD_MIN_RUN = 40               # 도로로 인정할 최소 가로 폭(px)
ROAD_MIN_FILL = 0.30            # 한 열이 도로로 인정되려면 밴드의 몇 배가 차야 하는지

ROAD_KP = 0.6            # 좌우 치우침 반영 정도
ROAD_KD = 0.35           # 곡률(앞 도로가 휘는 정도) 반영 정도
ROAD_GAIN = 0.5          # 조향을 바퀴 속도차로 얼마나 낼지 (0=직진만)
ROAD_MIN_STEER_TIME = 0.15 # 이보다 짧은 전진은 그냥 직진 (마지막 미세 접근)
# 0.4 였는데 0.15 로 내렸다.

# 조각 사이마다 모터를 껐다 켜면 주행 한 번에 80번쯤 서고 출발한다 = 눈에 보이는 딸꾹질.
# 조향은 조각마다 다시 주되 정지는 하지 않는다 (2026-08-29). True 로 두면 예전 거동.
# ★ 이걸 끄면 같은 시간에 더 멀리 간다. MOVE_FORWARD_PER_ONE 을 `road-cal` 로 다시
#   재기 전까지는 HOP 거리가 그만큼 길어지므로 반드시 재측정할 것.
# 2026-08-29: 경기장 바닥을 못 써서 road-cal 을 실제 매트에서 못 잰다. 재기 전에
# False 로 두면 HOP 이 통째로 오버슈트하므로, 검증 전까지는 True(예전 거동)로 둔다.
ROAD_STOP_BETWEEN_STEPS = True


def _otsu(ch, lo, hi):
    """이 프레임 안에서 밝은/어두운(또는 저채도/고채도) 경계를 스스로 찾는다."""
    t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(max(lo, min(hi, t)))


def road_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if ROAD_AUTO:
        s_max = _otsu(hsv[:, :, 1], 30, 120)     # 도로는 채도가 낮은 쪽
        v_min = _otsu(hsv[:, :, 2], 100, 220)    # 도로는 명도가 높은 쪽
    else:
        s_max, v_min = ROAD_S_MAX, ROAD_V_MIN
    mask = cv2.inRange(hsv, np.array([0, 0, v_min]),
                            np.array([179, s_max, 255]))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)      # 흰 점 노이즈 제거
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)     # 도로 위 글자 구멍 메우기


def road_band_center(mask, band):
    """밴드에서 도로 중심 x. 못 찾으면 None. 가장 넓은 '연속' 구간만 쓴다."""
    h, w = mask.shape
    y0, y1 = int(h * band[0]), int(h * band[1])
    sub = mask[y0:y1]
    if sub.size == 0:
        return None

    on = (sub.sum(axis=0) / 255) >= (y1 - y0) * ROAD_MIN_FILL
    if not on.any():
        return None

    best_len = best_start = 0
    cur = None
    for x, v in enumerate(np.append(on, False)):
        if v and cur is None:
            cur = x
        elif not v and cur is not None:
            if x - cur > best_len:
                best_len, best_start = x - cur, cur
            cur = None

    if best_len < ROAD_MIN_RUN:
        return None
    return best_start + best_len / 2


def road_band_width(mask, band):
    """밴드에서 가장 넓은 연속 도로 구간의 폭(px). 교차로 감지에 쓴다."""
    h, w = mask.shape
    y0, y1 = int(h * band[0]), int(h * band[1])
    sub = mask[y0:y1]
    if sub.size == 0:
        return 0
    on = (sub.sum(axis=0) / 255) >= (y1 - y0) * ROAD_MIN_FILL
    best = cur = 0
    for v in np.append(on, False):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def road_offset(frame):
    """(error, curve). 도로를 못 찾으면 (None, None). error : -1~+1. 도로 중심이 화면 중심보다 왼쪽이면 양수 = 로봇이 오른쪽으로 치우친 것 -> 왼쪽으로 꺾어야 한다."""
    mask = road_mask(frame)
    w = mask.shape[1]
    near = road_band_center(mask, ROAD_NEAR_BAND)
    far = road_band_center(mask, ROAD_FAR_BAND)
    if near is None and far is None:
        return None, None
    if near is None:
        near = far
    if far is None:
        far = near
    error = (w / 2 - near) / (w / 2)
    curve = (near - far) / (w / 2)
    return float(np.clip(error, -1, 1)), float(np.clip(curve, -1, 1))


def road_steer(frame):
    """조향량 -1(좌) ~ +1(우). 도로를 못 찾으면 None -> 직진 유지."""
    error, curve = road_offset(frame)
    if error is None:
        return None
    return float(np.clip(-(ROAD_KP * error + ROAD_KD * curve), -1, 1))


@_timed("도로주행")
def move_forward_on_road(duration_time, step=0.12, motor_speed=MOTOR_SPEED):
    """step 은 0.25 였는데 0.12 로 내렸다."""
    if not ROAD_FOLLOW:
        return move_forward(duration_time, motor_speed)
    """도로 중심을 보며 전진한다.

    긴 전진을 짧게 쪼개고, 매 조각 직전에 도로를 보고 좌우 바퀴 속도를 다르게 준다.
    원본은 목표까지 계산한 시간만큼 눈 감고 직진해서, 도로가 휘면 그대로 잔디로 나갔다.

    짧은 전진(마지막 미세 접근)은 조향하지 않는다. 이미 아루코로 정렬된 상태라
    거기서 또 꺾으면 정렬이 흐트러진다.
    """
    if duration_time < ROAD_MIN_STEER_TIME:
        move_forward(duration_time, motor_speed)
        return

    remaining = duration_time
    while remaining > 0:
        t = min(step, remaining)
        _fr = time.time()
        _frame = pinky_cam.get_frame()
        _timing["└ 그중 카메라대기"] = (_timing.get("└ 그중 카메라대기", 0.0)
                                       + time.time() - _fr)
        s = road_steer(_frame)
        if s is None:
            _log("  도로 안 보임 -> 직진")
            bias = 0
        else:
            bias = int(motor_speed * ROAD_GAIN * s)
        # 전진 중이므로 양쪽 다 앞으로 돌게 유지한다 (한쪽이 음수면 제자리 회전이 된다)
        left = max(10, min(100, motor_speed + bias))
        right = max(10, min(100, motor_speed - bias))
        pinky_motor.move(left, right)
        time.sleep(t)
        if ROAD_STOP_BETWEEN_STEPS:
            pinky_motor.move(0, 0)
        remaining -= t
    pinky_motor.move(0, 0)

# ================================================================ 아루코

@_timed("마커검출")
def detect_target_aruco(aruco_num):
    """한 프레임에서 목표 id 마커를 찾는다."""
    frame = pinky_cam.get_frame()
    output_frame, pose = pinky_cam.detect_aruco(frame, marker_size=MARKER_SIZE_M)

    if not pose:
        _log("None is detected")
        return False, None

    for p in pose:
        if int(p[0]) != aruco_num:
            continue
        z = p[3]
        # 거리로 한 번 거른다.
        if not (Z_ACCEPT_MIN <= z <= Z_ACCEPT_MAX):
            _log(f"Target id:{aruco_num} z:{z:.1f} — 거리 밖이라 무시 "
                  f"({Z_ACCEPT_MIN}~{Z_ACCEPT_MAX}cm 만 인정)")
            continue
        print(f"Target detected  id:{int(p[0])} x:{p[1]:.1f} y:{p[2]:.1f} z:{z:.1f}")
        return True, [p]              # 호출부가 pose[0][n] 을 쓰므로 리스트로 감싼다

    _log("Target not detected (보이는 id:", [int(p[0]) for p in pose], ")")
    return False, None


# direction : 0(Left), 1(Right)
def find_aruco_with_try_count(aruco_num, direction, try_count, deadline=None):
    """한 방향으로 훑으며 마커를 찾는다."""
    if try_count == 0:
        return False, None

    span = SEARCH_STEP_DEG * try_count
    rate = TURN_DEG_PER_SEC * SEARCH_SCAN_SPEED / MOTOR_SPEED   # 도/초
    chunk = SEARCH_STEP_DEG / rate                              # 한 스텝 회전 시간
    turn = move_left if direction == LEFT else move_right

    y0 = read_yaw()
    for i in range(try_count):
        if deadline and time.time() > deadline:
            _log("  탐색 중단 (시간 초과)")
            break
        success, pose = detect_target_aruco(aruco_num)
        if success:
            print("find_aruco, detected")
            return True, pose
        turn(chunk, SEARCH_SCAN_SPEED)
        _log(f"search {i + 1}/{try_count} ({SEARCH_STEP_DEG * (i + 1)}/{span}도)")

    # 마지막 한 번 더 본다 (마지막 회전 뒤를 안 보고 끝내면 손해다)
    success, pose = detect_target_aruco(aruco_num)
    if success:
        print("find_aruco, detected (마지막)")
        return True, pose

    # 실제로 돈 각도만큼 되돌린다
    y1 = read_yaw()
    if y0 is not None and y1 is not None:
        turned = (y1 - y0 + 180) % 360 - 180
        if abs(turned) > 1:
            turn_deg(-turned)
    else:
        back = move_right if direction == LEFT else move_left
        back(chunk * try_count, SEARCH_SCAN_SPEED)
    return False, None


def find_aruco_forever(aruco_num, direction):
    while True:
        success, pose = detect_target_aruco(aruco_num)
        time.sleep(0.3)
        if success:
            print("no limit, detected")
            return True, pose
        else:
            if direction == LEFT:
                turn_left_deg(SEARCH_STEP_DEG)
            else:
                turn_right_deg(SEARCH_STEP_DEG)
            time.sleep(SLEEP_TIME_AFTER_MOVE)


def find_aruco(aruco_num, direction, try_count, deadline=None):
    """지정 방향 -> 반대 방향으로 훑고, 그래도 없으면 조금 전진해서 다시 본다."""
    for attempt in range(FIND_ADVANCE_TRIES + 1):
        if deadline and time.time() > deadline:
            _log(f"  마커 {aruco_num} 탐색 시간 초과")
            return False, None
        result, pose = find_aruco_with_try_count(aruco_num, direction, try_count, deadline)
        if result:
            return True, pose

        time.sleep(SLEEP_TIME_AFTER_MOVE)
        other = RIGHT if direction == LEFT else LEFT
        result, pose = find_aruco_with_try_count(aruco_num, other, try_count, deadline)
        if result:
            return True, pose

        if attempt < FIND_ADVANCE_TRIES and not (deadline and time.time() > deadline):
            _log(f"  마커 {aruco_num} 안 보임 — {FIND_ADVANCE_CM}cm 전진 후 재탐색 "
                  f"({attempt + 1}/{FIND_ADVANCE_TRIES})")
            move_forward_on_road(FIND_ADVANCE_CM / MOVE_FORWARD_PER_ONE)
            time.sleep(SLEEP_TIME_AFTER_MOVE)
    return False, None

def check_angle(aruco_num, target, allow_range=15):
    """(맞았나, 돌아야 할 각도) 를 돌려준다."""
    success, pose = detect_target_aruco(aruco_num)
    if not success:
        _log("check_angle, not detected")
        return False, None

    cur_x, z = pose[0][1], pose[0][3]
    off = cur_x - target
    print(f"current_pose_x {cur_x:.1f} target_pose_x {target} (오차 {off:+.1f}cm)")
    if abs(off) <= allow_range:
        return True, 0.0

    # 오른쪽으로 치우쳐 있으면(off>0) 오른쪽으로 돌아야 한다 -> 양수
    deg = math.degrees(math.atan2(off, max(z, 1.0)))
    deg = max(-MATCH_MAX_DEG, min(MATCH_MAX_DEG, deg))
    print(f"  -> {deg:+.1f}도 회전")
    return False, deg


def check_distance(aruco_num, target, allow_range=5):
    """(도착했나, 전진할 시간, 지금 본 z) 를 돌려준다."""
    success, pose = detect_target_aruco(aruco_num)
    if not success:
        _log("check_distance, not detected")
        return False, None, None

    cur_pose_z = pose[0][3]
    print("cur_pose_z", round(cur_pose_z, 1), "target_pose_z", target)
    if cur_pose_z > target + allow_range:
        # 아직 멀다 -> 남은 거리를 시간으로 환산하되, 한 번에 가는 양을 제한한다
        temp_time = abs(target - cur_pose_z) / MOVE_FORWARD_PER_ONE
        capped = min(temp_time, APPROACH_MAX_STEP)
        extra = "" if capped >= temp_time else f" (전체 {temp_time:.2f}초 중)"
        print(f"남은 {cur_pose_z - target:.0f}cm -> {capped:.2f}초 전진{extra}")
        return False, capped, cur_pose_z
    # 범위 안이거나 지나쳤으면 도착으로 본다 (후진해서 되돌리지 않는다)
    return True, None, cur_pose_z


def track_target_aruco_marker(aruco_num, target_pose, try_count=0, timeout=MARKER_TIMEOUT):
    """1) 마커 탐색 -> 2) 각도(x) 정렬 -> 3) 거리(z) 접근. 원본 대비 바뀐 점 두 가지 (둘 다 안 고치면 대회에서 그대로 물린다): - 각도 정렬 루프에 탈출구가 없었다."""
    target_x, target_z, target_direction = target_pose
    deadline = time.time() + timeout

    ok, pose = find_aruco(aruco_num, target_direction, try_count, deadline)
    if not ok:
        _log("track_target_aruco_marker find_aruco_failed")
        return False

    # --- 각도(x) 맞추기 ---
    lost = 0
    while True:
        if time.time() > deadline:
            _log(f"[timeout] 마커 {aruco_num} 각도 정렬 {timeout}초 초과, 포기")
            return False

        aligned, angle_deg = check_angle(aruco_num, target_x)
        if aligned:
            break

        if angle_deg is None:                # 마커를 놓쳤다
            lost += 1
            _log("angle: 마커 놓침", lost)
            if lost >= 5:
                lost = 0
                ok, _ = find_aruco(aruco_num, target_direction, try_count, deadline)
                if not ok:
                    _log("angle: 재탐색 실패")
                    return False
            time.sleep(SLEEP_TIME_AFTER_MOVE)
            continue

        lost = 0
        if abs(angle_deg) < MATCH_MIN_DEG:   # 너무 작으면 돌아봐야 coast 에 묻힌다
            break
        turn_deg(angle_deg)
        time.sleep(SLEEP_TIME_AFTER_MOVE)
    print("angle success")

    # --- 거리(z) 맞추기 ---
    global _last_arrival_z
    _last_arrival_z = None
    not_detected_count = 0
    last_z, last_step = None, None
    while True:
        if time.time() > deadline:
            _log(f"[timeout] 마커 {aruco_num} 거리 접근 {timeout}초 초과, 여기서 멈춤")
            break

        arrived, distance_direction, z_now = check_distance(aruco_num, target_z)

        # 직전 스텝이 실제로 몇 cm 를 갔는지 관찰한다.
        if z_now is not None:
            if last_z is not None and last_step:
                moved = last_z - z_now
                print(f"    실측 전진 {moved:.0f}cm / {last_step:.2f}초 "
                      f"= {moved / last_step:.0f}cm/s (상수 {MOVE_FORWARD_PER_ONE:.0f})")
            last_z = z_now

        if arrived:
            _last_arrival_z = last_z
            globals()["_last_arrival_id"] = aruco_num
            print(f"도착 (마지막으로 본 거리 {last_z:.0f}cm)" if last_z else "도착")
            break

        if distance_direction is None:
            not_detected_count += 1
            _log("distance detection failed", not_detected_count)
            if not_detected_count >= 3:
                if last_z is not None:
                    _log(f"마커를 놓쳤다. 마지막으로 본 거리 {last_z:.0f}cm "
                          f"(목표 {target_z}cm)")
                break
            time.sleep(SLEEP_TIME_AFTER_MOVE)
            continue

        not_detected_count = 0
        last_step = distance_direction
        # 눈 감고 직진하면 도로가 휘는 구간에서 그대로 잔디로 나간다(패널티 5초/회).
        move_forward_on_road(distance_direction)
        time.sleep(SLEEP_TIME_AFTER_MOVE)

    print("distance success")
    return True

# ================================================================ 동작 실행기

def after_target_do_list(index):
    global _last_arrival_z
    global total_apple_count
    current_action = after_track_list[index]["actions"]
    if len(current_action) == 0:
        print("nothing")
        return

    for action_inside, option_inside in current_action:
        if action_inside == GO_STRAIGHT:
            # 0.4초 미만(과수원 코 들이밀기)은 안에서 그냥 직진으로 빠진다
            move_forward_on_road(option_inside)
        elif action_inside == MOVE_RIGHT:
            turn_right_deg(option_inside)      # 옵션 단위는 '도'
        elif action_inside == MOVE_LEFT:
            turn_left_deg(option_inside)
        elif action_inside == GO_BACKWARD:
            move_backward(option_inside)
        elif action_inside == APPLE_DISPLAY:
            total_apple_count += collect_apple_counts()
            display_apple_count(total_apple_count)
        elif action_inside == APPLE_COUNT_ACTION:
            start_apple_count()
            print(f"사진 {APPLE_CHECK_COUNT}장 촬영 — 서버 전송은 주행하면서 백그라운드로")
        elif action_inside == GO_TO_MARKER:
            # 마커를 잡았을 때 실제로 잰 거리를 쓴다.
            z = _last_arrival_z
            if z is None:
                _log("  마커 거리를 모른다 — 전진 생략")
            else:
                extra = 0 if option_inside == DEFAULT else option_inside
                done = target_list[index].get("hop") or 0
                remain = z - done + extra
                print(f"  마커까지 잰 {z:.0f}cm - 이미 간 {done}cm + {extra}cm "
                      f"= {remain:.0f}cm 전진")
                if remain > 0:
                    move_forward_on_road(remain / MOVE_FORWARD_PER_ONE)
                else:
                    # END_EXTRA_CM 이 음수라 도달 가능한 분기다.
                    _log(f"  !! 전진량이 {remain:.0f}cm — END 동작을 건너뛴다. "
                          f"마커10 을 너무 가깝게 쟀다(z={z:.0f})")
        elif action_inside == CHECK_NEXT:
            # 과수원을 보고 돌아오는 회전 중간에서 다음 마커를 본다.
            nxt = target_list[index + 1]["id"] if index + 1 < len(target_list) else None
            if nxt is not None:
                ok, seen = detect_target_aruco(nxt)
                if ok:
                    globals()["_last_arrival_z"] = seen[0][3]
                    globals()["_last_arrival_id"] = nxt
                    print(f"  다음 마커 {nxt} 확인 (x={seen[0][1]:+.0f} z={seen[0][3]:.0f})")
                else:
                    _log(f"  다음 마커 {nxt} 안 보임 — 외운 거리로 간다")
        elif action_inside == CROSS_WALK_WAIT:
            wait = 0.5 if option_inside == DEFAULT else option_inside
            print(f"횡단보도 {wait}초 대기")
            time.sleep(wait)
        else:
            print("wrong_action")
        time.sleep(SLEEP_TIME_AFTER_MOVE)

# ================================================================ 하드웨어
def make_run_dir():
    """주행마다 새 폴더. 이름은 날짜시간+그날 몇 번째인지, 숫자만."""
    os.makedirs(APPLE_SHOT_DIR, exist_ok=True)
    day = time.strftime("%Y%m%d")
    n = sum(1 for d in os.listdir(APPLE_SHOT_DIR) if d.startswith(day)) + 1
    path = f"{APPLE_SHOT_DIR}/{time.strftime('%Y%m%d%H%M')}{n:02d}"
    os.makedirs(path, exist_ok=True)
    print(f"사진 폴더 {path}")
    return path


def setup(motors=True):
    """카메라·모터·LCD 를 켠다. 주피터와 달리 프로세스가 끝나면 자동으로 반납된다."""
    global pinky_cam, pinky_motor, pinky_lcd, pinky_buzzer, pinky_imu, SERVER_URL
    from pinkylib import Camera, Motor, Buzzer
    from pinky_lcd.pinky_lcd import LCD

    pinky_cam = Camera()
    # set_calibration 의 기본값은 상대경로라 실행 위치에 따라 파일을 못 찾는다.
    pinky_cam.set_calibration(CALIBRATION_PATH)
    pinky_cam.start()

    if motors:                      # check/pose 로는 빈 폴더를 만들지 않는다
        globals()["run_dir"] = make_run_dir()
    pinky_lcd = LCD()
    if motors:
        pinky_motor, pinky_buzzer = Motor(), Buzzer()
        pinky_motor.enable_motor()
        pinky_buzzer.buzzer_start()
        try:
            from pinkylib import IMU
            pinky_imu = IMU()
            assert read_yaw() is not None
        except Exception as e:
            pinky_imu = None
            print(f"IMU 사용 불가 ({e}) — 회전을 시간 기반으로 폴백한다")

    SERVER_URL = get_server_url(my_ip)
    print("하드웨어 준비 완료")


def teardown():
    for dev in (pinky_cam, pinky_motor, pinky_lcd, pinky_buzzer, pinky_imu):
        try:
            if dev is not None:
                dev.close()
        except Exception as e:
            _log("close 실패:", e)


# ================================================================ 메인 주행
def run_course():
    global total_apple_count
    _cal.clear()
    assert len(target_list) == len(after_track_list), \
        "target_list 와 after_track_list 의 개수가 다릅니다"
    for t in target_list:
        x, _, d = t["pose"]
        # 마커가 왼쪽에 있는데 오른쪽부터 훑으면 120도를 헛돈다.
        assert x == 0 or (x < 0) == (d == LEFT), \
            f"마커 {t['id']}: x={x} 인데 탐색 방향이 반대다 (x<0 이면 LEFT)"

    start_time = time.time()
    for i in range(len(target_list)):
        current_id = target_list[i]["id"]
        print(f"\n===== [{i}] 마커 {current_id} 로 이동 "
              f"(경과 {time.time() - start_time:.0f}s) =====")

        # 맵을 외웠으니 대부분의 거리는 카메라를 안 보고 먼저 간다.
        cm = target_list[i].get("cm")
        if cm:
            blind = cm * APPROACH_BLIND_RATIO
            print(f"  외운 거리 {cm}cm 중 {blind:.0f}cm 를 먼저 이동")
            move_forward_on_road(blind / MOVE_FORWARD_PER_ONE)

        # 화각 끝에 걸린 마커는 잠깐 틀어서 확인만 하고 제자리로 돌아온다.
        if target_list[i].get("confirm"):
            peek = target_list[i].get("peek") or 0
            deg = -peek if target_list[i]["pose"][0] < 0 else peek
            if deg:
                turn_deg(deg)
            result, seen = detect_target_aruco(current_id)
            # GO_TO_MARKER 가 이 값을 쓴다.
            if result:
                globals()["_last_arrival_z"] = seen[0][3]
                globals()["_last_arrival_id"] = current_id
            elif _last_arrival_id != current_id:
                # 직전 구간의 CHECK_NEXT 가 이미 이 마커를 실측해 뒀으면 그게 더 정확하다.
                globals()["_last_arrival_z"] = target_list[i]["pose"][1]
                globals()["_last_arrival_id"] = current_id
            if result:
                print(f"  마커 {current_id} 확인 "
                      f"(x={seen[0][1]:+.0f} z={seen[0][3]:.0f})")
            else:
                _log(f"  마커 {current_id} 안 보임 — 외운 거리로 간다")
            if deg:
                turn_deg(-deg)          # 튼 만큼 되돌린다. 방향은 도로가 정한다
            _cal.append((current_id, cm, target_list[i]["pose"][1],
                         seen[0][3] if result else None, result))
            hop = target_list[i].get("hop") or 0
            if hop:
                print(f"  교차로까지 외운 {hop}cm 이동")
                # move_forward 로 바꿔봤다가 (2026-08-28, "진입 딜레이" 제거 목적) 도로
                # 보정 없이 눈 감고 가서 실주행에서 구동이 이상해져 도로 원래대로 되돌림.
                move_forward_on_road(hop / MOVE_FORWARD_PER_ONE)
            after_target_do_list(i)
            print(f"----list num : {i} done -----")
            continue

        result = track_target_aruco_marker(current_id, target_list[i]["pose"], SEARCH_COUNT)
        _cal.append((current_id, cm, target_list[i]["pose"][1], _last_arrival_z, result))
        if not result:
            # 평가는 "어디까지 갔나" 로 그룹이 갈린다 (메인도로 E < 교차로1 D < 과수원 C < 도착 B < 하차장 A). 마커 하나
            # 놓쳤다고 전체를 포기하면 그룹이 내려간다.
            _log(f"!! 마커 {current_id} 실패 — 남은 거리만 마저 가고 진행")
            if cm:
                move_forward_on_road(cm * (1 - APPROACH_BLIND_RATIO) / MOVE_FORWARD_PER_ONE)

        # 추적에 실패했어도 로봇은 이미 그 자리에 있다.
        hop = target_list[i].get("hop") or 0
        if result and hop:
            print(f"  마커 확정 -> 교차로까지 외운 {hop}cm 를 마저 간다")
            move_forward_on_road(hop / MOVE_FORWARD_PER_ONE)

        after_target_do_list(i)
        print(f"----list num : {i} done -----")

    pinky_motor.move(0, 0)
    print("\n===== 거리 캘리브레이션 =====")
    print(" 마커   cm   목표z    실제z   마커~교차로 여유")
    for mid, cm, tz, az, ok in _cal:
        if az is None:
            _log(f" {mid:>4}  {cm:>4}   {tz:>4}    ----   마커 실패 — cm 이 너무 길거나 짧다")
            continue
        # 이제 전부 confirm 이라 z 는 "구간 시작에서 마커까지" 다.
        hop_now = next(t["hop"] for t in target_list if t["id"] == mid)
        print(f" {mid:>4}  {cm:>4}   {tz:>4}   {az:5.1f}   "
              f"hop {hop_now} - z = gap {hop_now - az:+.1f}cm")
    total_apple_count += collect_apple_counts()   # 아직 안 거둔 게 있으면 여기서
    elapsed = time.time() - start_time
    print("\n===== 시간 분해 =====")
    inner = sum(v for k, v in _timing.items() if not k.startswith("└"))
    for k, v in sorted(_timing.items(), key=lambda kv: -kv[1]):
        print(f" {k:>14s} {v:6.2f}초 ({v / elapsed * 100:4.1f}%)")
    print(f" {'설명 안 되는 몫':>14s} {elapsed - inner:6.2f}초 "
          f"({(elapsed - inner) / elapsed * 100:4.1f}%)")
    print(f"\n===== 주행 종료. 사과 {total_apple_count}개 / {elapsed:.0f}초 =====")
    display_apple_count(total_apple_count)


# ================================================================ 서브커맨드


def wait_for_enter(msg="엔터를 누르면 시작"):
    """하드웨어 준비를 먼저 끝내고 엔터에서 출발한다."""
    try:
        input(f"\n>>> {msg} (Ctrl+C 로 취소) ")
    except EOFError:
        # 터미널이 아니면(ssh -t 없이 실행) 그냥 진행한다
        print("  (입력 없음 — 바로 시작)")


def cmd_run(*args):
    setup()
    try:
        print(f"\n준비 완료 — 마커 {[t['id'] for t in target_list]} 순서로 주행")
        _, pose = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=MARKER_SIZE_M)
        print("지금 보이는 마커:", [int(p[0]) for p in pose] if pose else "없음",
              "/ 첫 목표", target_list[0]["id"])
        if "now" not in args:
            wait_for_enter("엔터를 누르면 주행 시작")
        run_course()
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n중단됨")
        pinky_motor.move(0, 0)
    finally:
        teardown()


def cmd_check():
    """출발 전 점검. 모터를 켜지 않으므로 로봇이 움직이지 않는다."""
    setup(motors=False)
    ok = True
    try:
        print("OK   캘리브레이션" if pinky_cam.calibration_matrix is not None
              else "FAIL 캘리브레이션 — CALIBRATION_PATH 확인")
        ok &= pinky_cam.calibration_matrix is not None

        if my_ip == "192.168.x.x":
            print("FAIL my_ip 미입력 — drive.py 의 my_ip 를 PC IP 로"); ok = False
        else:
            try:
                frame = pinky_cam.get_frame()
                r = requests.post(get_server_url(my_ip), timeout=10, files={
                    "image": ("t.jpg", cv2.imencode(".jpg", frame)[1].tobytes(), "image/jpeg")})
                print(f"OK   서버 {r.status_code} · 사과 {r.json().get('detected_count')}개")
            except Exception as e:
                print("FAIL 서버 —", e, "/ run.bat, PC 방화벽, my_ip 확인"); ok = False

        if len(target_list) != len(after_track_list):
            print("FAIL target_list / after_track_list 개수 불일치"); ok = False
        else:
            print(f"OK   리스트 {len(target_list)}개")

        frame = pinky_cam.get_frame()
        _, pose = pinky_cam.detect_aruco(frame, marker_size=MARKER_SIZE_M)
        print("     보이는 마커:", [int(p[0]) for p in pose] if pose else "없음",
              "/ 첫 목표", target_list[0]["id"])

        e, c = road_offset(frame)
        if e is None:
            print("FAIL 도로 인식 실패 — 'drive.py road' 로 임계값 확인"); ok = False
        else:
            print(f"OK   도로 error={e:+.2f} curve={c:+.2f} steer={road_steer(frame):+.2f}")

        print("\n" + ("=== 출발 가능 ===" if ok else "=== 위 FAIL 부터 고칠 것 ==="))
    finally:
        teardown()
    return 0 if ok else 1


def cmd_pose():
    """마커 앞에 로봇을 세워두고 실행. 출력 줄을 target_list 에 그대로 붙여넣는다."""
    setup(motors=False)
    try:
        _, pose = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=MARKER_SIZE_M)
        if not pose:
            print("마커가 안 보인다. 방향이나 거리를 조정할 것")
            return 1
        for p in pose:
            print(f'    {{"id": {int(p[0])}, "pose": [{p[1]:.0f}, {p[3]:.0f}, RIGHT]}},')
    finally:
        teardown()
    return 0


def cmd_forward_cal():
    """마커가 정면에 보이는 자리에 두고 실행. MOVE_FORWARD_PER_ONE 을 구한다."""
    setup()
    try:
        def z():
            _, p = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=MARKER_SIZE_M)
            assert p, "마커가 안 보인다"
            return p[0][3]
        z1 = z()
        move_forward(1)
        time.sleep(0.5)
        z2 = z()
        print(f"z {z1:.1f} -> {z2:.1f}")
        print(f"MOVE_FORWARD_PER_ONE = {abs(z2 - z1):.3f}   <- drive.py 에 입력")
    finally:
        teardown()
    return 0


def cmd_road_cal():
    """도로 추종 모드의 실제 전진 속도. forward-cal(연속 주행)과 다를 수 있다.

    forward-cal 은 move_forward 로 1초 쭉 가서 재는데, 실제 주행은 전부
    move_forward_on_road 를 쓴다. 조각내기 때문에 그 둘의 속도가 다르고,
    HOP 거리는 이 값으로 시간이 환산된다. 그러니 여기서 잰 값을 써야 맞다.

    마커가 정면에 보이는 자리에 두고 실행한다.
    """
    setup()
    try:
        def z():
            _, p = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=MARKER_SIZE_M)
            assert p, "마커가 안 보인다"
            return p[0][3]
        print(f"조각 사이 정지: {'있음(예전 거동)' if ROAD_STOP_BETWEEN_STEPS else '없음'}")
        z1 = z()
        move_forward_on_road(1.0)
        time.sleep(0.5)
        z2 = z()
        print(f"z {z1:.1f} -> {z2:.1f}")
        print(f"MOVE_FORWARD_PER_ONE = {abs(z2 - z1):.3f}   <- drive.py 에 입력")
    finally:
        teardown()
    return 0


def cmd_turn_cal():
    """IMU 로 닫은 회전이 실제로 맞는지 확인한다."""
    setup()
    try:
        print("IMU:", "사용" if pinky_imu else "없음(시간 폴백)")
        print(f"coast 모델: {TURN_COAST_PER_SPEED} x 속도 (도)\n")
        for deg in (90, -90, 180):
            input(f"엔터 -> {deg:+}도 회전 (원위치에 놓고) ")
            actual = turn_deg(deg)
            print(f"  목표 {deg:+}도 / IMU {actual:+.1f}도 -> 눈으로 몇 도인가?")

        input("\n엔터 -> 90도 x 4 (제자리로 돌아와야 한다) ")
        y0 = read_yaw()
        prev, total = y0, 0.0
        for i in range(4):
            turn_deg(90)
            time.sleep(0.4)
            y = read_yaw()
            if y is not None:
                total += (y - prev + 180) % 360 - 180
                prev = y
        print(f"\n누적 {total:.1f}도 / 360도 -> 오차 {total - 360:+.1f}도")
        print("로봇이 처음 방향으로 돌아왔으면 그대로 쓰면 된다.")
        print("계속 지나치면 TURN_COAST_PER_SPEED 를 키우고, 모자라면 줄인다.")
    finally:
        teardown()
    return 0


def cmd_road(path=None):
    """도로 마스크 확인. 사진 경로를 주면 그 사진으로, 없으면 지금 카메라로."""
    if path:
        frame = cv2.imread(path)
        assert frame is not None, path
    else:
        setup(motors=False)
        frame = pinky_cam.get_frame()
        teardown()

    e, c = road_offset(frame)
    print(f"error={e} curve={c} steer={road_steer(frame)}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = hsv[int(frame.shape[0] * 0.8):].reshape(-1, 3)
    print(f"아래 20%  S 중앙값 {np.median(low[:, 1]):.0f}  V 중앙값 {np.median(low[:, 2]):.0f}")
    print(f"추천  ROAD_S_MAX = {np.percentile(low[:, 1], 60) + 15:.0f}  "
          f"ROAD_V_MIN = {max(0, np.percentile(low[:, 2], 30) - 15):.0f}")
    print(f"현재  ROAD_S_MAX = {ROAD_S_MAX}  ROAD_V_MIN = {ROAD_V_MIN}")

    vis = frame.copy()
    vis[road_mask(frame) > 0] = (0, 0, 255)
    cv2.imwrite("road_check.jpg", vis)
    print("road_check.jpg 저장 (빨강 = 도로로 인식한 영역)")
    return 0


ACTION_NAMES = {GO_STRAIGHT: "전진", MOVE_RIGHT: "우회전", MOVE_LEFT: "좌회전",
                GO_BACKWARD: "후진", APPLE_COUNT_ACTION: "사과 세기",
                GO_TO_MARKER: "마커까지 전진",
                CHECK_NEXT: "다음 마커 확인",
                APPLE_DISPLAY: "LCD 표시", CROSS_WALK_WAIT: "대기"}


def cmd_motors():
    """모터 방향 확인. 좌우가 바뀌어 있으면 코스 전체가 반대로 간다 — 제일 먼저 볼 것."""
    setup()
    try:
        for name, fn, arg, unit in (("전진", move_forward, 0.5, "초"),
                                    ("후진", move_backward, 0.5, "초"),
                                    ("좌회전", turn_left_deg, 90, "도"),
                                    ("우회전", turn_right_deg, 90, "도")):
            input(f"엔터 -> {name} {arg}{unit} ")
            fn(arg)
            print(f"  {name} 완료 — 실제로 그렇게 움직였는지 확인")
        print("\n하나라도 반대면 drive.py 의 move_* 를 고치지 말고 배선을 확인할 것")
    finally:
        teardown()
    return 0


def cmd_track(marker_id, target_z=None):
    """마커 하나만 찾아가서 선다."""
    marker_id = int(marker_id)
    setup()
    try:
        pose = [0, float(target_z) if target_z else 20, RIGHT]
        print(f"마커 {marker_id} 추적 (목표 x={pose[0]} z={pose[1]}cm)")
        wait_for_enter("엔터를 누르면 추적 시작")
        t0 = time.time()
        ok = track_target_aruco_marker(marker_id, pose, SEARCH_COUNT)
        pinky_motor.move(0, 0)
        print(f"{'성공' if ok else '실패'} · {time.time() - t0:.0f}초")

        _, seen = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=MARKER_SIZE_M)
        for pp in seen or []:
            if int(pp[0]) == marker_id:
                print(f"최종 위치: x={pp[1]:.1f} z={pp[3]:.1f}cm  "
                      f"<- 이 값을 target_list 에 넣으면 된다")
        return 0 if ok else 1
    except KeyboardInterrupt:
        print("\n중단됨")
        pinky_motor.move(0, 0)
        return 1
    finally:
        teardown()


def cmd_actions(index):
    """after_track_list 의 한 항목만 실행해 본다 (회전 방향·PEEK 시간 확인용)."""
    index = int(index)
    setup()
    try:
        print(f"[{index}] 마커 {after_track_list[index]['id']} 의 동작:")
        for a, o in after_track_list[index]["actions"]:
            print(f"    {ACTION_NAMES.get(a, a)}  {o}")
        wait_for_enter("엔터를 누르면 동작 실행")
        after_target_do_list(index)
        pinky_motor.move(0, 0)
    except KeyboardInterrupt:
        print("\n중단됨")
        pinky_motor.move(0, 0)
    finally:
        teardown()
    return 0


TELEOP_PORT = 8080
TELEOP_SPEED = 40          # 텔레옵 전진 속도. 주행(90)보다 느려야 세우기 쉽다
TELEOP_TURN = 32           # 제자리 회전 속도
TELEOP_DEADMAN = 0.4       # 이 시간 안에 다음 명령이 안 오면 선다


def cmd_teleop(port=TELEOP_PORT):
    """브라우저로 로봇을 몰면서 카메라와 아루코 인식을 실시간으로 본다."""
    import http.server
    import urllib.parse

    setup()
    state = {"lr": (0, 0), "t": 0.0, "stop": False}

    def deadman():
        while not state["stop"]:
            if state["lr"] != (0, 0) and time.time() - state["t"] > TELEOP_DEADMAN:
                state["lr"] = (0, 0)
                pinky_motor.move(0, 0)
            time.sleep(0.05)

    KEYS = {"w": (TELEOP_SPEED, TELEOP_SPEED),
            "s": (-TELEOP_SPEED, -TELEOP_SPEED),
            "a": (-TELEOP_TURN, TELEOP_TURN),
            "d": (TELEOP_TURN, -TELEOP_TURN),
            "q": (TELEOP_SPEED // 2, TELEOP_SPEED),
            "e": (TELEOP_SPEED, TELEOP_SPEED // 2),
            " ": (0, 0)}

    def draw(frame):
        vis = frame.copy()
        h, w = vis.shape[:2]
        cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        _, pose = pinky_cam.detect_aruco(frame, marker_size=MARKER_SIZE_M)
        for i, p in enumerate(pose or []):
            mid, x, z = int(p[0]), p[1], p[3]
            d = "RIGHT" if x >= 0 else "LEFT "
            cv2.putText(vis, f'{{"id": {mid}, "pose": [{x:+.0f}, {z:.0f}, {d}]}},',
                        (8, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 255, 0), 2)
        if not pose:
            cv2.putText(vis, "marker: none", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 160, 255), 2)
        return vis

    PAGE = b"""<!doctype html><meta charset=utf-8><title>WCRC teleop</title>
<style>body{background:#111;color:#eee;font:14px monospace;text-align:center;margin:0}
img{max-width:100%;image-rendering:pixelated}
b{color:#6f6}</style>
<img src=/stream><div id=s>W/S \xec\x95\x9e\xeb\x92\xa4 &nbsp; A/D \xed\x9a\x8c\xec\xa0\x84 &nbsp;
Q/E \xec\x99\x84\xeb\xa7\x8c\xed\x95\x9c \xec\xa2\x8c\xec\x9a\xb0 &nbsp;
\xea\xb0\x84\xea\xb2\xa9 \xec\xa0\x95\xec\xa7\x80 &nbsp; <b id=k>-</b></div>
<script>
let cur='';
function send(k){cur=k;fetch('/k?k='+encodeURIComponent(k));document.getElementById('k').textContent=k===' '?'stop':k}
setInterval(()=>{if(cur&&cur!==' ')fetch('/k?k='+encodeURIComponent(cur))},150);
addEventListener('keydown',e=>{const k=e.key.toLowerCase();
 if('wsadqe '.includes(k)&&k!==cur){e.preventDefault();send(k)}});
addEventListener('keyup',e=>{if('wsadqe'.includes(e.key.toLowerCase()))send(' ')});
addEventListener('blur',()=>send(' '));
</script>"""

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            path, _, q = self.path.partition("?")
            if path == "/k":
                k = urllib.parse.parse_qs(q).get("k", [" "])[0]
                lr = KEYS.get(k, (0, 0))
                state["lr"], state["t"] = lr, time.time()
                pinky_motor.move(*lr)
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path != "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    jpg = cv2.imencode(".jpg", draw(pinky_cam.get_frame()))[1].tobytes()
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=deadman, daemon=True).start()
    print(f"http://192.168.4.1:{port}   (Ctrl-C 로 종료)")
    print("화면의 초록 줄을 그대로 target_list 에 붙이면 된다.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state["stop"] = True
        srv.server_close()
        teardown()
    return 0


CMDS = {"run": cmd_run, "check": cmd_check, "pose": cmd_pose,
        "motors": cmd_motors, "track": cmd_track, "actions": cmd_actions,
        "forward-cal": cmd_forward_cal, "road-cal": cmd_road_cal, "turn-cal": cmd_turn_cal,
        "road": cmd_road, "teleop": cmd_teleop}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd not in CMDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(CMDS[cmd](*sys.argv[2:]) or 0)
