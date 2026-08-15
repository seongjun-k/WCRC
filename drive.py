#!/usr/bin/env python3
"""2026 WCRC 주행 코드 (Pinky pro). 주피터 없이 로봇에서 바로 돌린다.

    ssh pinky@192.168.4.1
    python3 drive.py check          # 출발 전 점검 (모터 안 돎)
    python3 drive.py run            # ★ 실제 주행 (준비 후 엔터에서 출발)
    python3 drive.py run now        # 엔터 없이 바로 출발

단계별 테스트 (코스 전체를 돌리기 전에 이 순서로):
    python3 drive.py motors         # 1. 모터 방향 (좌우 바뀌면 전부 반대로 간다)
    python3 drive.py track 1        # 2. 마커 하나만 찾아가 보기
    python3 drive.py actions 0      # 3. 마커 하나의 도착 후 동작만
    python3 drive.py run            # 4. 전체 주행

측정용 서브커맨드 (코스에서 값 채울 때):
    python3 drive.py pose           # 지금 보이는 마커의 x, z 출력 -> target_list 채우기
    python3 drive.py forward-cal    # MOVE_FORWARD_PER_ONE 측정
    python3 drive.py turn-cal       # IMU 회전 확인 (90도x4 로 제자리 검증)
    python3 drive.py road [사진]    # 도로 마스크가 지금 화면에서 되는지 확인

주피터를 안 쓰는 이유: 브라우저를 닫아도 커널이 남아 카메라·모터를 물고 있어서
"카메라를 찾을 수 없습니다" 가 뜬다. 스크립트는 끝나면서 하드웨어를 반납한다.

PC 쪽 검증:  python tools/sim_drive.py   /   python tools/road.py
둘 다 이 파일을 그대로 읽어서 돌린다. 여기를 고치면 같이 검증된다.
"""
import os
import sys
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# 하드웨어 모듈은 setup() 안에서 늦게 import 한다.
# PC 에서 검증 도구가 이 파일을 import 할 수 있어야 하기 때문 (PC 엔 pinkylib 이 없다).
pinky_cam = pinky_motor = pinky_lcd = pinky_buzzer = pinky_imu = None
SERVER_URL = None

# ================================================================ 상수

LEFT = 0
RIGHT = 1
FORWARD = 2

# after_track_list 에서 사용하는 동작 코드
GO_STRAIGHT = 100         # 전진
MOVE_RIGHT = 101          # 우회전
MOVE_LEFT = 102           # 좌회전
GO_BACKWARD = 103         # 후진
APPLE_COUNT_ACTION = 104  # 사과 개수 세기 (flask 서버로 이미지 전송, 개수 응답받기)
APPLE_DISPLAY = 105       # LCD에 사과 개수 표시하기
CROSS_WALK_WAIT = 106     # 잠시 대기 (횡단보도). 이제 초 단위 인자를 받는다
DEFAULT = 999             # 시간 옵션이 필요 없는 동작에 사용


MOTOR_SPEED = 90
SEARCH_MOTOR_SPEED = 65

# 제자리 회전은 시간이 아니라 **각도**로 지정하고, IMU(BNO055)로 실제 각도를 보며 닫는다.
# 시간으로 돌리면 배터리·바닥 마찰·바퀴 미끄러짐에 그대로 흔들린다. 실측에서
# 엔코더 틱은 시간에 완벽히 선형인데(4212틱/초) 실제 회전각은 그보다 덜 나왔다 —
# 차이가 바로 바퀴 미끄러짐이고, 이건 모터 쪽 상수로는 못 잡는다.
# 정지 명령을 내려도 관성으로 더 돈다(coast). 실측하면 속도에 정확히 비례한다:
#     속도 15 -> 3.2도   25 -> 5.7도   35 -> 8.0도   50 -> 10.9도   60 -> 12.7도
# 즉 coast(도) = 0.211 x 속도. 그래서 속도를 고정하면 그 coast 보다 작은 각도는
# 아예 만들 수 없다. 속도 60 으로 6도를 돌라고 하면 17도가 돈다.
# -> 목표 각도에 맞춰 **속도를 고른다**. 작은 각도는 느리게 돈다.
TURN_COAST_PER_SPEED = 0.211   # ★실측
TURN_COAST_RATIO = 0.30        # coast 가 목표각의 이 비율을 넘지 않게 속도를 정한다
TURN_MIN_SPEED = 10            # 이 아래로는 안 내린다 (실측상 10 에서도 잘 돈다)
TURN_MAX_SPEED = 60
TURN_TOL_DEG = 1.5         # 이 안에 들어오면 도달로 본다 (넘으면 한 번 보정)
TURN_MAX_SEC = 4.0         # IMU 가 이상해도 여기서 끊는다

# IMU 를 못 쓸 때의 폴백. 실측: MOTOR_SPEED 90 에서 212도/초, 데드타임 없음(선형).
TURN_DEG_PER_SEC = 212

SEARCH_STEP_DEG = 15       # 마커를 찾으며 한 번에 도는 각도
MATCH_STEP_DEG = 6         # 각도 정렬에서 한 번에 도는 각도
MATCH_FORWARD_TIME = 0.4

SLEEP_TIME_AFTER_MOVE = 0.15
MOTOR_BIG_STEP_FORWARD = 1

STRAIGHT_TO_MAIN_ROAD_TIME = 3  # 모터 스피드에 따라 변경 필요할 수 있음

# 아루코 마커 하나에 매달릴 최대 시간(초).
# 규정 패널티 1번 "모든 단계에서 60초 이상 진행 없을 시 기회 종료".
# 제자리에서 마커를 찾아 도는 건 심판 눈에 '진행 없음'으로 보이기 쉬우므로 60초에
# 붙이지 않고 35초에서 포기하고 다음 마커로 넘어간다.
# (양방향 탐색 최악이 약 22초 + 정렬 여유 13초)
MARKER_TIMEOUT = 35

# 과수원·하차장을 "살짝 들여다보는" 전진/후진 시간(초).
# 가지 안까지 들어가지 않는다. 90도 돌아 코만 들이밀고 세고 그대로 후진해서 뺀다.
# 전진과 후진을 같은 값으로 둬야 원래 자리로 돌아온다.
PEEK_TIME = 0.3

# 하차장에서 나온 뒤 횡단보도 STOP 라인까지 전진할 시간(초).
# 횡단보도에는 아루코 마커가 없다. 표지판을 볼 필요도 없다 — 하차장 진입/복귀가
# 매번 같은 자리라 여기서 출발하는 지점도 재현되기 때문이다.
# 침범해도 보너스가 -20초에서 -10초로 줄 뿐이니, 애매하면 짧게 잡는 쪽이 낫다.
TO_CROSSWALK_TIME = 2.0    # 약 100cm (50cm/s 기준)  ★코스에서 조정

# set_calibration() 은 기본값이 상대경로("camera_calibration.npz")라 노트북 위치에 따라
# FileNotFoundError 가 난다. 로봇에 실제로 파일이 있는 절대경로를 박아둔다.
CALIBRATION_PATH = "/home/pinky/CH/camera_calibration.npz"


# 사과 갯수가 담기는 전역 변수
# (규정 3번: 잘 익은 사과 = 빨간 사과만 카운팅. 모델도 빨간 사과만 학습되어 있어야 한다)
total_apple_count = 0

# ---------------------------------------------------------------- 사용자 설정

# SEARCH_COUNT: 아루코 마커를 찾으려고 한 방향으로 몇 번 조금씩 회전할지.
# 기본값 5는 5 x SEARCH_TURN_TIME(0.05초) = 0.25초 회전 = 사실상 제자리다.
# 과수원에서 돌아나온 뒤처럼 마커가 시야 밖에 있을 때 절대 못 찾는다.
# 한 스텝이 SEARCH_STEP_DEG(15도)이므로 12스텝이면 한 방향 180도, 양방향 360도.
# 이보다 키우면 같은 자리를 두 번 훑는다.
SEARCH_COUNT = 12
APPLE_CHECK_COUNT = 3     # 사과 이미지 flask 서버에 보낼 횟수 (기본값 사용)

# ★ [학생 수정 ①] my_ip — Flask 서버 PC의 IP 주소

my_ip = "192.168.4.7"     # ← PC IP. 로봇 AP 에 붙은 PC 주소. `ip a` / `ipconfig` 로 확인

# ★ [학생 수정 ②] MOVE_FORWARD_PER_ONE — 직접 측정!

MOVE_FORWARD_PER_ONE = 50.067   # cm/초. `drive.py forward-cal` 실측값 (2026-08-15)

# 한 번에 전진할 최대 시간(초). 남은 거리를 한 방에 가면 그 사이 아무것도 못 보고,
# MOVE_FORWARD_PER_ONE 이 틀린 만큼 그대로 지나친다(마커를 놓쳐 도착 확인도 못 한다).
# 잘라서 가며 매번 다시 재면 이 상수가 좀 틀려도 알아서 수렴한다 — 회전을 IMU 로
# 닫은 것과 같은 이유다.
APPROACH_MAX_STEP = 0.5

# 탐색에 실패했을 때 원래 방향으로 되돌아올 각도. 훑은 만큼 그대로 되돌린다.
SEARCH_RETURN_DEG = SEARCH_STEP_DEG * SEARCH_COUNT

# ---------------------------------------------------------------- 주행 순서

# ★ [학생 수정 ③] 아루코마커 주행 순서 id 리스트
#
# 맵(26wcrc_final.pdf 3p) 기준:
#   START → 교차로1(1) → 교차로2(2) → 교차로3(4) → 교차로4(5) → 횡단보도 → 도착점(10) → END
#
# ★ 과수원 마커(3,6,9)와 하차장 마커(0)는 목표에서 뺐다.
#   가지 안까지 들어가지 않고, 교차로 마커에 정렬한 그 자리에서 90도 돌아 코만 들이밀고
#   세고 후진해서 빠진다(after_track_list 참고). 규정 2번의 "진입했다는 유의미한 행위" 는
#   회전 + 전진으로 충족된다. 이렇게 하면
#     - 180도 회전이 통째로 없어진다 (이 코스에서 제일 안 맞는 동작)
#     - 후진이라 복귀 후 방향이 안 틀어진다
#     - 찾아야 할 마커가 9개 → 5개로 줄어 실패 지점이 준다
#
# pose = [target_x, target_z, 탐색_시작_방향]   단위는 cm
#
# ★★ 아래 x, z 는 전부 "형식만 맞춘 초기값"이다. `drive.py pose` 로 실측해 교체할 것.
#
# 횡단보도에는 아루코 마커가 없다. 하차장에서 나온 뒤 정해진 시간만큼 가서 선다
# (마커5 의 동작 참고). 하차장 진입·복귀가 매번 같은 자리라 출발 지점도 재현된다.
target_list = [
    {"id": 1,  "pose": [0, 20, RIGHT]},   # 교차로1 (여기서 과수원A 를 들여다본다)  ★실측
    {"id": 2,  "pose": [0, 20, RIGHT]},   # 교차로2 (과수원B)  ★실측
    {"id": 4,  "pose": [0, 20, RIGHT]},   # 교차로3 (과수원C)  ★실측
    {"id": 5,  "pose": [0, 20, RIGHT]},   # 교차로4 (하차장 + 횡단보도)  ★실측
    {"id": 10, "pose": [0, 40, RIGHT]},   # 도착점 END  ★실측
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
                           (MOVE_LEFT, 90)]},

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

    # 교차로4 → 하차장 (북쪽): 왼쪽. LCD 에 누적 개수 표시 (규정 6번 변경분: 부저 아님).
    # 그 뒤 돌아나와 횡단보도까지 정해진 시간만큼 가서 3초 정지 (규정 7·8번, 보너스 -20초)
    {"id": 5,  "actions": [(MOVE_LEFT, 90),
                           (GO_STRAIGHT, PEEK_TIME),
                           (APPLE_DISPLAY, DEFAULT),
                           (GO_BACKWARD, PEEK_TIME),
                           (MOVE_RIGHT, 90),
                           (GO_STRAIGHT, TO_CROSSWALK_TIME),
                           (CROSS_WALK_WAIT, 3.0)]},

    # 도착점 END: 정지만
    {"id": 10, "actions": []},
]


# ================================================================ 서버 통신

def get_server_url(ip):
    return f"http://{ip}:5000/predict"

def send_image_and_get_count(image_input):  # 이미지를 PC(Flask)로 전송하고 감지된 사물의 총 개수(count)를 받아옵니다.
    try:
        # 1) 파일 경로(문자열)인 경우
        if isinstance(image_input, str):
            with open(image_input, 'rb') as f:
                files = {'image': f}
                response = requests.post(SERVER_URL, files=files)

        # 2) Picamera2 / OpenCV frame (Numpy Array)인 경우
        else:
            # 메모리 상에서 즉시 JPEG 바이너리로 인코딩 (파일 저장 없이 고속 처리)
            success, img_encoded = cv2.imencode('.jpg', image_input)
            if not success:
                print(" image encoding failed")
                return None

            files = {'image': ('robot_frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post(SERVER_URL, files=files)

        # 응답 처리
        if response.status_code == 200:
            res_data = response.json()

            # [디버그 출력] 서버가 실제로 보내온 데이터 전체를 확인합니다.
            print("Server respose raw data:", res_data)

            # 서버에서 'detected_count' 키값을 가져옴
            if 'detected_count' in res_data:
                count = res_data['detected_count']
            else:
                print("warning:'detected_count' key is missing in Server response")
                count = 0

            print(f"detected count: {count} (file name: {res_data.get('saved_filename')})")
            return count

        elif response.status_code == 400:
            err_msg = response.json().get('message', '요청 에러')
            print(f" Error : {err_msg}")
            return None

        else:
            print(f" Server status error (code {response.status_code}):", response.text)
            return None

    except Exception as e:
        print(f"Communication exception :", e)
        return None

# ================================================================ LCD / 사과

def display_apple_count(apple_count):
    global total_apple_count
    img_width, img_height = 320, 240
    background_color = (0, 0, 0)

    img = Image.new('RGB', (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)

    text_color = (255, 255, 255)
    text = f"Total apple count : {total_apple_count}"

    try:
        font = ImageFont.truetype("NanumGothic.ttf", 30)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img_width - text_width) // 2
    y = (img_height - text_height) // 2
    draw.text((x, y), text, fill=text_color, font=font)

    pinky_lcd.img_show(img)


def predict_apple_count():
    # APPLE_CHECK_COUNT 번 촬영해서 서버로 보내고, 그중 가장 큰 값을 사용합니다.
    temp_count = 0
    for i in range(APPLE_CHECK_COUNT):
        frame = pinky_cam.get_frame()
        count = send_image_and_get_count(frame)
        if count is not None:
            if count > temp_count:
                temp_count = count
    return temp_count

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
    """IMU 의 yaw(도). 못 읽으면 None.

    6-DOF 모드라 자이로 적분값이다. 수 초 단위의 상대 회전에는 충분하고,
    우리가 쓰는 건 전부 "지금부터 몇 도" 라서 장시간 드리프트는 문제되지 않는다.
    """
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
    pinky_motor.move(speed * sign, -speed * sign)
    while abs(turned) < target and time.time() < deadline:
        y = read_yaw()
        if y is not None:
            turned += (y - prev + 180) % 360 - 180   # 360도 넘김 처리
            prev = y
        time.sleep(0.01)
    pinky_motor.move(0, 0)

    time.sleep(0.35)                     # 멈춘 뒤 실제로 더 돈 만큼까지 읽는다
    y = read_yaw()
    if y is not None:
        turned += (y - prev + 180) % 360 - 180
    return turned


def turn_deg(deg):
    """제자리에서 deg 도 돈다 (양수 = 우회전). 실제로 돈 각도를 돌려준다.

    IMU 로 닫으므로 배터리 잔량·바닥 재질·바퀴 미끄러짐과 무관하게 같은 각도가 나온다.
    한 번 돌고 남은 오차가 크면 한 번만 더 보정한다 (계속 보정하면 진동한다).
    IMU 를 못 읽으면 시간 기반으로 폴백한다.
    """
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


# ▼ 대회장 조명에서 재조정할 값 2개 (tools/road.py --tune 이 추천해준다)
ROAD_S_MAX = 70          # 이보다 채도가 높으면 도로가 아니다 (잔디·테두리·집기)
ROAD_V_MIN = 150         # 이보다 어두우면 도로가 아니다 (그림자)

ROAD_NEAR_BAND = (0.80, 1.00)   # 바로 앞  — 좌우 치우침 계산용
ROAD_FAR_BAND = (0.60, 0.78)    # 조금 먼 앞 — 도로가 휘는 방향 계산용
ROAD_MIN_RUN = 40               # 도로로 인정할 최소 가로 폭(px)
ROAD_MIN_FILL = 0.30            # 한 열이 도로로 인정되려면 밴드의 몇 배가 차야 하는지

ROAD_KP = 0.6            # 좌우 치우침 반영 정도
ROAD_KD = 0.35           # 곡률(앞 도로가 휘는 정도) 반영 정도
ROAD_GAIN = 0.5          # 조향을 바퀴 속도차로 얼마나 낼지 (0=직진만)
ROAD_MIN_STEER_TIME = 0.4  # 이보다 짧은 전진은 그냥 직진 (마지막 미세 접근)


def road_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, ROAD_V_MIN]),
                            np.array([179, ROAD_S_MAX, 255]))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)      # 흰 점 노이즈 제거
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)     # 도로 위 글자 구멍 메우기


def road_band_center(mask, band):
    """밴드에서 도로 중심 x. 못 찾으면 None.

    가장 넓은 '연속' 구간만 쓴다. 화면에 흰 물체가 여럿일 때 전체 평균을 내면
    엉뚱한 가운데가 나오지만, 연속 구간을 쓰면 실제 도로 면을 고른다.
    """
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


def road_offset(frame):
    """(error, curve). 도로를 못 찾으면 (None, None).

    error : -1~+1. 도로 중심이 화면 중심보다 왼쪽이면 양수
            = 로봇이 오른쪽으로 치우친 것 -> 왼쪽으로 꺾어야 한다.
    curve : -1~+1. 앞 도로가 휘는 방향.
    """
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


def move_forward_on_road(duration_time, step=0.25, motor_speed=MOTOR_SPEED):
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
        s = road_steer(pinky_cam.get_frame())
        if s is None:
            print("  도로 안 보임 -> 직진")
            bias = 0
        else:
            bias = int(motor_speed * ROAD_GAIN * s)
        # 전진 중이므로 양쪽 다 앞으로 돌게 유지한다 (한쪽이 음수면 제자리 회전이 된다)
        left = max(10, min(100, motor_speed + bias))
        right = max(10, min(100, motor_speed - bias))
        pinky_motor.move(left, right)
        time.sleep(t)
        pinky_motor.move(0, 0)
        remaining -= t

# ================================================================ 아루코

def detect_target_aruco(aruco_num):
    """한 프레임에서 목표 id 마커를 찾는다.

    pinkylib 의 detect_aruco 는 화면에 보이는 마커 '전부'를 [id, x, y, z] 리스트로 준다.
    원본 코드는 pose[0] 하나만 보고 id 를 비교해서, 교차로처럼 마커가 두 개 이상
    잡히는 자리에서 목표가 두 번째면 영원히 "not detected" 가 됐다. 전체를 훑는다.
    """
    frame = pinky_cam.get_frame()
    output_frame, pose = pinky_cam.detect_aruco(frame, marker_size=0.1)

    if not pose:
        print("None is detected")
        return False, None

    for p in pose:
        if int(p[0]) == aruco_num:
            print(f"Target detected  id:{int(p[0])} x:{p[1]:.1f} y:{p[2]:.1f} z:{p[3]:.1f}")
            return True, [p]          # 호출부가 pose[0][n] 을 쓰므로 리스트로 감싼다

    print("Target not detected (보이는 id:", [int(p[0]) for p in pose], ")")
    return False, None


# direction : 0(Left), 1(Right)
def find_aruco_with_try_count(aruco_num, direction, try_count):
    if try_count != 0:
        for i in range(try_count):
            success, pose = detect_target_aruco(aruco_num)
            time.sleep(0.3)
            if success:
                print("find_aruco, detected")
                return True, pose
            else:
                if direction == LEFT:
                    turn_left_deg(SEARCH_STEP_DEG)
                else:
                    turn_right_deg(SEARCH_STEP_DEG)
                print("search step", i)
                time.sleep(SLEEP_TIME_AFTER_MOVE)
        # try_count 번 다 돌아도 못 찾으면 원래 방향으로 복귀.
        # ★ 반드시 탐색과 같은 속도(SEARCH_MOTOR_SPEED)로 돌려야 한다.
        #   원본은 속도 인자를 안 넘겨서 기본값 MOTOR_SPEED(90)로 복귀했고,
        #   같은 시간을 1.4배 속도로 도니 탐색할수록 로봇 방향이 틀어졌다.
        if direction == LEFT:
            turn_right_deg(SEARCH_RETURN_DEG)
        else:
            turn_left_deg(SEARCH_RETURN_DEG)
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


def find_aruco(aruco_num, direction, try_count):
    # 지정한 방향으로 try_count 번 탐색하고, 실패하면 반대 방향으로 한 번 더 탐색합니다.
    result, pose = find_aruco_with_try_count(aruco_num, direction, try_count)
    if result:
        return True, pose

    time.sleep(SLEEP_TIME_AFTER_MOVE)
    other = RIGHT if direction == LEFT else LEFT
    return find_aruco_with_try_count(aruco_num, other, try_count)

def check_angle(aruco_num, target, allow_range=15):
    success, pose = detect_target_aruco(aruco_num)
    if not success:
        print("check_angle, not detected")
        return False, None

    cur_pose_x = pose[0][1]
    print("current_pose_x", round(cur_pose_x, 1), "target_pose_x", target)
    if cur_pose_x < target - allow_range:
        return False, LEFT
    if cur_pose_x > target + allow_range:
        return False, RIGHT
    return True, 0          # 경계값에서 아무 분기도 안 타고 None 이 반환되던 구멍을 막았다


def check_distance(aruco_num, target, allow_range=5):
    success, pose = detect_target_aruco(aruco_num)
    if not success:
        print("check_distance, not detected")
        return False, None

    cur_pose_z = pose[0][3]
    print("cur_pose_z", round(cur_pose_z, 1), "target_pose_z", target)
    if cur_pose_z > target + allow_range:
        # 아직 멀다 -> 남은 거리를 시간으로 환산하되, 한 번에 가는 양을 제한한다
        temp_time = abs(target - cur_pose_z) / MOVE_FORWARD_PER_ONE
        capped = min(temp_time, APPROACH_MAX_STEP)
        extra = "" if capped >= temp_time else f" (전체 {temp_time:.2f}초 중)"
        print(f"남은 {cur_pose_z - target:.0f}cm -> {capped:.2f}초 전진{extra}")
        return False, capped
    # 범위 안이거나 지나쳤으면 도착으로 본다 (후진해서 되돌리지 않는다)
    return True, None


def track_target_aruco_marker(aruco_num, target_pose, try_count=0, timeout=MARKER_TIMEOUT):
    """1) 마커 탐색 -> 2) 각도(x) 정렬 -> 3) 거리(z) 접근.

    원본 대비 바뀐 점 두 가지 (둘 다 안 고치면 대회에서 그대로 물린다):
      - 각도 정렬 루프에 탈출구가 없었다. check_angle 이 마커를 놓치면 (False, None) 을
        주는데 받는 쪽이 `if LEFT ... else move_right` 라서 None 이 전부 우회전으로 떨어져
        영원히 제자리 우회전을 했다. → 규정 패널티 1번(60초 정체 시 기회 종료) 직행.
      - 마커 하나에 무한정 매달렸다. → timeout 초를 넘기면 포기하고 False 를 돌려준다.
    """
    target_x, target_z, target_direction = target_pose
    deadline = time.time() + timeout

    ok, pose = find_aruco(aruco_num, target_direction, try_count)
    if not ok:
        print("track_target_aruco_marker find_aruco_failed")
        return False

    # --- 각도(x) 맞추기 ---
    lost = 0
    while True:
        if time.time() > deadline:
            print(f"[timeout] 마커 {aruco_num} 각도 정렬 {timeout}초 초과, 포기")
            return False

        aligned, angle_direction = check_angle(aruco_num, target_x)
        if aligned:
            break

        if angle_direction is None:          # 마커를 놓쳤다
            lost += 1
            print("angle: 마커 놓침", lost)
            if lost >= 5:
                lost = 0
                ok, _ = find_aruco(aruco_num, target_direction, try_count)
                if not ok:
                    print("angle: 재탐색 실패")
                    return False
            time.sleep(SLEEP_TIME_AFTER_MOVE)
            continue

        lost = 0
        if angle_direction == LEFT:
            turn_left_deg(MATCH_STEP_DEG)
        else:
            turn_right_deg(MATCH_STEP_DEG)
        time.sleep(SLEEP_TIME_AFTER_MOVE)
    print("angle success")

    # --- 거리(z) 맞추기 ---
    not_detected_count = 0
    last_z, last_step = None, None
    while True:
        if time.time() > deadline:
            print(f"[timeout] 마커 {aruco_num} 거리 접근 {timeout}초 초과, 여기서 멈춤")
            break

        arrived, distance_direction = check_distance(aruco_num, target_z)

        # 직전 스텝이 실제로 몇 cm 를 갔는지 관찰한다. 이 값이 상수와 크게 다르면
        # forward-cal 로 MOVE_FORWARD_PER_ONE 을 다시 재야 한다는 뜻이다.
        ok, pose_now = detect_target_aruco(aruco_num)
        if ok:
            if last_z is not None and last_step:
                moved = last_z - pose_now[0][3]
                print(f"    실측 전진 {moved:.0f}cm / {last_step:.2f}초 "
                      f"= {moved / last_step:.0f}cm/s (상수 {MOVE_FORWARD_PER_ONE:.0f})")
            last_z = pose_now[0][3]

        if arrived:
            print(f"도착 (마지막으로 본 거리 {last_z:.0f}cm)" if last_z else "도착")
            break

        if distance_direction is None:
            not_detected_count += 1
            print("distance detection failed", not_detected_count)
            if not_detected_count >= 3:
                if last_z is not None:
                    print(f"마커를 놓쳤다. 마지막으로 본 거리 {last_z:.0f}cm "
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
            display_apple_count(total_apple_count)
        elif action_inside == APPLE_COUNT_ACTION:
            temp_apple_count = predict_apple_count()
            total_apple_count = total_apple_count + temp_apple_count
            print(f"이번 과수원 {temp_apple_count}개 / 누적 {total_apple_count}개")
        elif action_inside == CROSS_WALK_WAIT:
            # 원본은 option 을 무시하고 0.5초 고정이었다. 규정 8번은 3초라 인자를 쓴다.
            wait = 0.5 if option_inside == DEFAULT else option_inside
            print(f"횡단보도 {wait}초 대기")
            time.sleep(wait)
        else:
            print("wrong_action")
        time.sleep(SLEEP_TIME_AFTER_MOVE)

# ================================================================ 하드웨어
def setup(motors=True):
    """카메라·모터·LCD 를 켠다. 주피터와 달리 프로세스가 끝나면 자동으로 반납된다."""
    global pinky_cam, pinky_motor, pinky_lcd, pinky_buzzer, pinky_imu, SERVER_URL
    from pinkylib import Camera, Motor, Buzzer
    from pinky_lcd.pinky_lcd import LCD

    pinky_cam = Camera()
    # set_calibration 의 기본값은 상대경로라 실행 위치에 따라 파일을 못 찾는다.
    # 여기서 실패하면 아루코가 통째로 안 잡히고 주행이 아예 안 된다.
    pinky_cam.set_calibration(CALIBRATION_PATH)
    pinky_cam.start()

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
            print("close 실패:", e)


# ================================================================ 메인 주행
def run_course():
    assert len(target_list) == len(after_track_list), \
        "target_list 와 after_track_list 의 개수가 다릅니다"

    start_time = time.time()
    for i in range(len(target_list)):
        current_id = target_list[i]["id"]
        print(f"\n===== [{i}] 마커 {current_id} 로 이동 "
              f"(경과 {time.time() - start_time:.0f}s) =====")

        result = track_target_aruco_marker(current_id, target_list[i]["pose"], SEARCH_COUNT)
        if not result:
            # 평가는 "어디까지 갔나" 로 그룹이 갈린다 (메인도로 E < 교차로1 D < 과수원 C
            # < 도착 B < 하차장 A). 마커 하나 놓쳤다고 전체를 포기하면 그룹이 내려간다.
            print(f"!! 마커 {current_id} 실패 — 그래도 다음 동작을 하고 진행한다")

        # 추적에 실패했어도 로봇은 이미 그 자리에 있다. 복귀 동작을 안 하면 갇히므로
        # 성공/실패와 무관하게 실행한다.
        after_target_do_list(i)
        print(f"----list num : {i} done -----")

    pinky_motor.move(0, 0)
    print(f"\n===== 주행 종료. 사과 {total_apple_count}개 / "
          f"{time.time() - start_time:.0f}초 =====")
    display_apple_count(total_apple_count)


# ================================================================ 서브커맨드
def wait_for_enter(msg="엔터를 누르면 시작"):
    """하드웨어 준비를 먼저 끝내고 엔터에서 출발한다.

    규정 1번이 "심판이 시작 신호 보낸 후 코드 실행 / 엔터 또는 클릭 후 키보드 접촉 금지"
    이고, 신호 후 60초 안에 못 나가면 기회가 줄어든다. 카메라 초기화에 몇 초 걸리므로
    미리 켜두고 엔터만 기다리는 게 맞다.
    """
    try:
        input(f"\n>>> {msg} (Ctrl+C 로 취소) ")
    except EOFError:
        # 터미널이 아니면(ssh -t 없이 실행) 그냥 진행한다
        print("  (입력 없음 — 바로 시작)")


def cmd_run(*args):
    setup()
    try:
        print(f"\n준비 완료 — 마커 {[t['id'] for t in target_list]} 순서로 주행")
        _, pose = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=0.1)
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
        _, pose = pinky_cam.detect_aruco(frame, marker_size=0.1)
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
        _, pose = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=0.1)
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
            _, p = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=0.1)
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


def cmd_turn_cal():
    """IMU 로 닫은 회전이 실제로 맞는지 확인한다.

    각도기 없이 검증하려면 90도를 네 번 돌려 제자리로 오는지 보면 된다.
    오차가 4배로 보인다.
    """
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
    """마커 하나만 찾아가서 선다. 코스 전체를 돌리기 전에 이걸로 먼저 확인한다.

        python3 drive.py track 1        # 마커 1 을 기본 거리까지
        python3 drive.py track 1 20     # 20cm 앞에서 정지
    """
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

        _, seen = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=0.1)
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
    """after_track_list 의 한 항목만 실행해 본다 (회전 방향·PEEK 시간 확인용).

        python3 drive.py actions 0      # 교차로1 의 '과수원A 들여다보기'
    """
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


CMDS = {"run": cmd_run, "check": cmd_check, "pose": cmd_pose,
        "motors": cmd_motors, "track": cmd_track, "actions": cmd_actions,
        "forward-cal": cmd_forward_cal, "turn-cal": cmd_turn_cal,
        "road": cmd_road}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd not in CMDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(CMDS[cmd](*sys.argv[2:]) or 0)
