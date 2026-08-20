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
import math
import os
import re
import shutil
import sys
import threading
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
GO_TO_MARKER = 107        # 마커를 잡았을 때 잰 z 만큼 전진 (옵션 = 더 갈 cm)
CROSS_WALK_WAIT = 106     # 잠시 대기 (횡단보도). 이제 초 단위 인자를 받는다
DEFAULT = 999             # 시간 옵션이 필요 없는 동작에 사용


MOTOR_SPEED = 90

# 좌우 모터 편향 보정. 왼쪽에 더하고 오른쪽에서 빼는 값(속도 단위, 양수 = 왼쪽을 빠르게).
# 무게가 좌우로 치우치면(센서 탈착·배터리 위치) 같은 속도를 줘도 직진이 휜다. 이건
# 상수 외란이라 도로 추종의 P 제어로는 못 없앤다 — P 는 오차에 비례해서만 밀어내므로
# 편향과 힘이 같아지는 지점, 즉 도로 한쪽에 붙은 채로 평형을 이룬다. 좌우 여유가
# 2.1cm 뿐이라 그대로 선을 넘고, 교차로에서 돌 때 마커를 친다.
# `python3 drive.py cal forward` 가 IMU 로 직진 중 yaw 가 도는 양을 재서 갱신한다.
MOTOR_TRIM = 0
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
TURN_COAST_PER_SPEED = 0.228   # ★실측. 0.211 로 돌려보니 회전당 +1.0도 초과 -> 보정
TURN_COAST_RATIO = 0.30        # coast 가 목표각의 이 비율을 넘지 않게 속도를 정한다
TURN_MIN_SPEED = 10            # 이 아래로는 안 내린다 (실측상 10 에서도 잘 돈다)
TURN_MAX_SPEED = 60
# 회전을 멈춘 뒤 coast 가 끝나길 기다리는 시간. 예전엔 0.35초 고정이었는데,
# coast 각도가 속도에 비례(0.211*speed)하므로 coast 시간도 속도에 비례한다.
# 90도 본회전(speed 60)은 그대로 0.34초, 오차 보정회전(speed 10~15)은 0.15초로 끝난다.
# 회전이 8번이라 보정회전 대기만 줄여도 랩타임이 준다.
TURN_SETTLE_BASE = 0.10
TURN_SETTLE_PER_SPEED = 0.004
TURN_TOL_DEG = 1.5         # 이 안에 들어오면 도달로 본다 (넘으면 한 번 보정)
TURN_MAX_SEC = 4.0         # IMU 가 이상해도 여기서 끊는다

# IMU 를 못 쓸 때의 폴백. 실측: MOTOR_SPEED 90 에서 212도/초, 데드타임 없음(선형).
TURN_DEG_PER_SEC = 212

SEARCH_STEP_DEG = 6        # 탐색 한 스텝의 회전 각도. 대기가 없어져 잘게 돌 수 있다
SEARCH_SCAN_SPEED = 25     # 탐색 회전 속도. 낮을수록 흔들림이 적고 덜 지나친다
MATCH_MAX_DEG = 30         # 정렬 회전 1회 상한. 계산이 틀렸을 때 크게 안 틀어지게
MATCH_MIN_DEG = 2          # 이보다 작으면 coast 에 묻히므로 돌지 않는다         # 각도 정렬에서 한 번에 도는 각도
MATCH_FORWARD_TIME = 0.4

SLEEP_TIME_AFTER_MOVE = 0.08
MOTOR_BIG_STEP_FORWARD = 1

STRAIGHT_TO_MAIN_ROAD_TIME = 3  # 모터 스피드에 따라 변경 필요할 수 있음

# 아루코 마커 하나에 매달릴 최대 시간(초).
# 규정 패널티 1번 "모든 단계에서 60초 이상 진행 없을 시 기회 종료".
# 제자리에서 마커를 찾아 도는 건 심판 눈에 '진행 없음'으로 보이기 쉬우므로 60초에
# 붙이지 않고 35초에서 포기하고 다음 마커로 넘어간다.
# (양방향 탐색 최악이 약 22초 + 정렬 여유 13초)
# 이 거리 밖의 "인식" 은 믿지 않는다. 실측 인식 구간이 37~75cm 라
# 그 밖에서 잡혔다는 건 멀리 있는 다른 마커를 잘못 읽었다는 뜻이다.
Z_ACCEPT_MIN = 8
Z_ACCEPT_MAX = 85

MARKER_TIMEOUT = 35

# 과수원·하차장을 "살짝 들여다보는" 전진/후진 시간(초).
# 가지 안까지 들어가지 않는다. 90도 돌아 코만 들이밀고 세고 그대로 후진해서 뺀다.
# 전진과 후진을 같은 값으로 둬야 원래 자리로 돌아온다.
PEEK_TIME = 0.28           # 17.72cm/s 로 5.0cm 전진 (과수원 진입 깊이)
# 5cm 는 실측으로 정한 값이다: 나무 앞에서 0cm 일 때는 사과가 작아 0개였고,
# 5.3cm 들어가자 1개로 잡혔다. 더 깊이 갈 이유가 없다 — 가지가 28~34cm 라
# 깊이 들어갈수록 후진 거리도 늘고 랩타임만 먹는다.

# 하차장에서 나온 뒤 횡단보도 STOP 라인까지 전진할 시간(초).
# 횡단보도에는 아루코 마커가 없다. 표지판을 볼 필요도 없다 — 하차장 진입/복귀가
# 매번 같은 자리라 여기서 출발하는 지점도 재현되기 때문이다.
# 침범해도 보너스가 -20초에서 -10초로 줄 뿐이니, 애매하면 짧게 잡는 쪽이 낫다.
# 정지선 -> END 는 고정 시간으로 못 간다. 마커10 을 잡는 위치가 앞 구간 오차만큼
# 흔들리기 때문이다 (실측 두 번: z=29.5cm, 38.6cm — 9cm 차이).
# 그래서 "잰 z 만큼 간다" 로 바꿨다. z 만큼 가면 마커10 과 나란히 서고 거기가 END 다.
END_EXTRA_CM = 18          # 실주행에서 1초(=17.7cm) 모자랐다. 선을 밟게 더 간다

# set_calibration() 은 기본값이 상대경로("camera_calibration.npz")라 노트북 위치에 따라
# FileNotFoundError 가 난다. 로봇에 실제로 파일이 있는 절대경로를 박아둔다.
# ★★ 아루코 마커 실물 한 변 길이(미터). OpenCV 관례라 단위는 미터다.
# 이 값이 틀리면 pinkylib 가 주는 z(거리, cm)가 통째로 `0.1/실제` 배로 어긋난다.
# 값만 어긋나는 게 아니라 아래가 전부 같이 틀어진다:
#   - target_list 의 z=20 ("20cm 앞에 정지")
#   - MOVE_FORWARD_PER_ONE (이 z 로 재서 구한 값이라 같은 배율로 부풀려져 있다)
# 그래서 이 값을 고치면 forward-cal 을 반드시 다시 돌려야 한다.
MARKER_SIZE_M = 0.036      # 실측 36mm x 36mm (2026-08-18)

# 마커를 도로 중심선에서 옆으로 얼마나 떼어 세울지(cm). 도로 반폭이 7.7cm,
# 로봇 바퀴 반폭이 5.55cm 이므로 12cm 면 풀밭 위이고 바퀴에 안 닿는다.
# 부호는 OpenCV 카메라 좌표계 기준 +가 오른쪽이다.
# ★ `drive.py pose` 로 실제 부호를 한 번 확인할 것 (오른쪽에 두고 x 가 +인가).
MARKER_SIDE_OFFSET = 12
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
SEARCH_COUNT = 10          # 탐색 스텝 수. 10 x 6도 = 60도.
# 예전엔 12(=180도)였다. 마커가 어느 쪽에 있는지 이미 아는데(pose 의 x 부호)
# 반 바퀴를 훑는 건 낭비고, 그만큼 로봇 방향도 틀어진다.
# 지정한 방향으로 60도 훑고, 없으면 반대로 60도 훑는다. 그래도 없으면 거리 문제이므로
# find_aruco 가 앞으로 조금 가서 다시 본다.
APPLE_CHECK_COUNT = 5     # 서버로 보낼 장수. 중앙값을 쓰므로 홀수로 둔다
APPLE_SHOT_GAP = 0.08     # 3장 사이 간격. 같은 프레임 3장이면 최댓값이 의미 없다
APPLE_JOIN_TIMEOUT = 6.0  # 서버가 죽어도 여기서 더 안 기다린다
APPLE_SHOT_DIR = "/home/pinky/apple_shots"   # 서버로 보낸 사진을 여기 남긴다

# ★ [학생 수정 ①] my_ip — Flask 서버 PC의 IP 주소

my_ip = "192.168.4.7"     # ← PC IP. 로봇 AP 에 붙은 PC 주소. `ip a` / `ipconfig` 로 확인

# ★ [학생 수정 ②] MOVE_FORWARD_PER_ONE — 직접 측정!

MOVE_FORWARD_PER_ONE = 17.72    # cm/초. 예열 후 4회 왕복 실측 중앙값 (2026-08-18)
# 냉간에는 13.2 cm/s 로 26% 느리다 -> 출발 전 warmup_motors() 로 예열한다.
# (예전 50.067 은 marker_size 를 0.1 로 잘못 두고 잰 값이라 2.78 배 부풀려져 있었다)

# 한 번에 전진할 최대 시간(초). 남은 거리를 한 방에 가면 그 사이 아무것도 못 보고,
# MOVE_FORWARD_PER_ONE 이 틀린 만큼 그대로 지나친다(마커를 놓쳐 도착 확인도 못 한다).
# 잘라서 가며 매번 다시 재면 이 상수가 좀 틀려도 알아서 수렴한다 — 회전을 IMU 로
# 닫은 것과 같은 이유다.
APPROACH_MAX_STEP = 0.5

# 탐색에 실패했을 때 원래 방향으로 되돌아올 각도. 훑은 만큼 그대로 되돌린다.
SEARCH_RETURN_DEG = SEARCH_STEP_DEG * SEARCH_COUNT

# 마커가 안 보일 때 제자리 회전만으로 안 풀리면 조금 전진해서 다시 본다.
# 36mm 마커는 75cm 밖에서 아예 안 잡히므로, 그 경우는 각도가 아니라 거리 문제다.
FIND_ADVANCE_CM = 8
FIND_ADVANCE_TRIES = 3

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
# cm 은 map.png(2100x1100mm) 도면에서 도로 마스크 최단경로로 실측한 값이다.
# 메인도로 전체가 START->END 220cm 밖에 안 된다. 구간이 36~48cm 로 짧다는 게
# 이 코스의 핵심 제약이다 — 회전 한 번 어긋나면 만회할 거리가 없다.
#
# z=45 는 "마커에서 45cm 앞에 선다"는 뜻이다. 20 이 아니라 45 인 이유:
#
#   36mm 마커의 실측 인식 구간이 37cm ~ 75cm 다 (그 밖은 10프레임 중 0회).
#   - 가까운 쪽 한계는 크기가 아니라 초점이다. OV5647 은 고정초점이라 37cm 아래로
#     들어가면 흐려져서 5x5 격자를 못 읽는다. 화면에 커다랗게 보여도 못 읽는다.
#   - 먼 쪽 한계 75cm 는 픽셀이 모자라서다.
#
# 그래서 45cm 로 잡았다 (구간 한가운데). 여기서 중요한 귀결:
#   구간이 최대 43cm 이므로 출발 지점에서 마커까지는 43+45=88cm 라 "안 보인다".
#   외운 거리로 먼저 달린 뒤에야 마커가 보인다 -> blind 선행은 최적화가 아니라 필수다.
#
# ★ 마커 위치는 우리가 못 정한다 — 도면의 노란 "마커N" 라벨 자리에만 놓을 수 있고,
#   그 자리는 교차로 바로 옆(12~20cm)이다. 여기서 두 가지가 동시에 걸린다:
#     - 36mm 마커는 37cm 아래로 들어오면 초점이 안 맞아 못 읽는다
#     - 교차로에 도착하면 마커는 옆으로 12~20cm, 즉 인식 한계 한참 안쪽이다
#   그래서 "마커 앞 N cm 에 정지" 라는 방식 자체가 성립하지 않는다.
#
#   대신 2단계로 간다:
#     1) 마커가 보이는 동안(z 37~75cm) 접근해서 위치를 한 번 확정한다  -> pose 의 z
#     2) 거기서 교차로까지 남은 거리는 외운 값으로 눈 감고 간다        -> "hop"
#
#   hop 은 실주행 실측으로 정했다 (2026-08-18). 규칙은 hop = 마커 고정 시의 실제 z.
#   마커가 교차로 옆에 붙어 있으므로, 마커까지 남은 깊이만큼 더 가면 교차로다.
#     마커1 z=21.2 -> hop 21 (hop 23 으로 완주 성공한 값과 일치. 이게 근거다)
#     마커2 z=35.6 -> 36    마커4 z=14.1 -> 14    마커5 z=16.7 -> 17
#   도면에서 계산했던 값(23/44/27/26)은 마커가 도면 라벨 중심에 있다고 본 것이라
#   실제와 최대 13cm 어긋났다. 실측이 우선이다.
#
#   pose = [target_x, target_z, 탐색_시작_방향]
#   ★ 탐색 방향은 target_x 의 부호와 같아야 한다 (x<0 이면 LEFT, x>0 이면 RIGHT).
#     마커가 왼-오-왼-오-왼 순으로 놓여 있으므로 그쪽으로 먼저 훑으면 몇 도만에 잡힌다.
#   target_x 는 0 이 아니라 그 지점에서의 실제 좌우 오프셋이다. 0 으로 두면
#   마커를 화면 한가운데 놓으려고 로봇이 도로에서 20도쯤 틀어진다.
#
target_list = [
    # 위치는 도면의 노란 라벨 영역 안에서 "hop 이 최소가 되는" 자리를 계산해 골랐다.
    # z 는 인식 한계 37cm 에 3cm 여유를 준 40 (마커10 은 화각 제약이라 46).
    # id  pose=[x, z, 탐색방향]        cm=직전 정지점부터   hop=마커 확정 후 눈감고
    {"id": 1,  "pose": [-17, 40, LEFT ], "cm": 26, "hop": 21},   # 교차로1 (과수원A)   마커 왼쪽
    {"id": 2,  "pose": [+17, 40, RIGHT], "cm":  7, "hop": 36},   # 교차로2 (과수원B)   마커 오른쪽
    {"id": 4,  "pose": [ -4, 40, LEFT ], "cm": 26, "hop": 14},   # 교차로3 (과수원C)   마커 왼쪽
    {"id": 5,  "pose": [+17, 40, RIGHT], "cm": 19, "hop": 17},   # 교차로4 (하차장)    마커 오른쪽
    # 마커2 의 cm 가 0 인 이유: 교차로1 에서 교차로2 까지가 43cm 인데 마커2 를
    # 44cm 앞에서 잡아야 해서, 교차로1 에 선 순간 이미 마커2 가 보인다.
    #
    # 마커10 은 교차로4 에서 9cm 간 지점에서 잡히고(hop 19), 그 hop 이 끝나는 곳이
    # 횡단보도 정지선이다. 그래서 정지선 정지를 마커10 의 동작으로 옮겼다.
    {"id": 10, "pose": [-21, 46, LEFT ], "cm":  9, "hop": 13},   # 횡단보도 정지선     마커 왼쪽
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
]},

    # 횡단보도 정지선에 섰다. 3초 서고 END 까지 간다.
    # END 는 "선을 밟기만" 하면 되므로 조금 지나쳐도 된다. 모자라는 게 더 나쁘다.
    {"id": 10, "actions": [(CROSS_WALK_WAIT, 3.0),
                           (GO_TO_MARKER, END_EXTRA_CM)]},
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

# LCD 폰트. NanumGothic.ttf 는 로봇에 없어서 원본 코드는 늘 기본 비트맵 폰트로
# 떨어졌고, 그래서 글씨가 아주 작게 나왔다. DejaVu 는 설치돼 있다.
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
    """규정 6번: 센 사과 개수를 LCD 에 표시한다.

    심판이 몇 걸음 떨어져서 봐야 하므로 숫자를 화면 가득 키운다.
    설명 문구는 위쪽에 작게만 둔다 — 읽어야 하는 건 숫자다.
    """
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


# 주행할 때마다 마커별 "목표 z vs 실제로 멈춘 z" 를 모은다. 주행 자체가
# 거리 캘리브레이션이 되게 하려는 것 — 따로 측정 모드를 돌릴 시간이 없다.
_cal = []
_last_arrival_z = None
_last_arrival_x = None   # 도착 시점에 마커가 옆으로 몇 cm 떨어져 있었나
_last_seen_x = None

# 제자리 회전 때 로봇 몸체가 쓸고 지나가는 반경(cm). 바퀴 폭 절반 5.55cm 에
# 앞뒤로 튀어나온 부분을 더한 값이다. ★실측할 것: 종이 위에 놓고 360도 돌린 뒤
# 자국의 반지름을 잰다. 규정 5번(지형지물 충돌) 패널티 5초가 여기에 걸린다.
ROBOT_SWEEP_CM = 8.0

_apple_jobs = []


def start_apple_count():
    """사진만 찍어두고 서버 전송은 백그라운드로 넘긴다.

    원래는 여기서 서버 왕복 3번을 기다렸다. 그 사이 로봇은 과수원 앞에 가만히
    서 있는데, 어차피 다음에 할 일(후진 + 90도 복귀)은 서버 응답과 무관하다.
    랩타임이 순위를 가르므로 겹칠 수 있는 건 겹친다.
    결과는 LCD 표시 직전(collect_apple_counts)에 한 번만 거둔다.
    """
    frames = []
    for i in range(APPLE_CHECK_COUNT):
        frames.append(pinky_cam.get_frame())
        if i < APPLE_CHECK_COUNT - 1:
            time.sleep(APPLE_SHOT_GAP)   # 완전히 같은 프레임 3장은 최댓값의 의미가 없다
    # 주행이 끝난 뒤 "로봇이 뭘 보고 셌나" 를 확인할 수 있어야 한다.
    # 개수가 0 으로 나왔을 때 원인이 카메라 방향인지 모델인지 이걸로 갈린다.
    n = len(_apple_jobs) + 1
    for i, f in enumerate(frames):
        try:
            cv2.imwrite(f"{APPLE_SHOT_DIR}/apple_{n}_{i}.jpg", f)
        except Exception as e:
            print("  사진 저장 실패:", e)
    print(f"  사진 {len(frames)}장 저장 -> {APPLE_SHOT_DIR}/apple_{n}_*.jpg")
    box = {}

    def work():
        # 최댓값이 아니라 중앙값을 쓴다.
        # 원본 템플릿은 max 였는데, 그건 "놓치는" 쪽이 문제일 때 맞는 선택이다.
        # 실측해보니 21프레임 중 20장이 정답 1개, 1장이 오탐 2개였다 (재현율 0.98 /
        # 정밀도 0.96). max 를 쓰면 그 한 장이 그대로 답이 되어 개수가 틀린다.
        # 중앙값은 한 장이 튀어도 흔들리지 않고, 반대로 한 장을 놓쳐도 버틴다.
        cs = [c for c in (send_image_and_get_count(f) for f in frames) if c is not None]
        box["n"] = sorted(cs)[len(cs) // 2] if cs else 0

    th = threading.Thread(target=work, daemon=True)
    th.start()
    _apple_jobs.append((th, box))


def collect_apple_counts():
    """백그라운드로 보낸 사과 개수를 전부 거둬 합계를 돌려준다."""
    total = 0
    for th, box in _apple_jobs:
        th.join(timeout=APPLE_JOIN_TIMEOUT)
        if th.is_alive():
            print("  [경고] 사과 개수 서버 응답 없음 — 그 과수원은 0개로 친다")
        total += box.get("n", 0)
    _apple_jobs.clear()
    return total

# ================================================================ 모터

def move_forward(duration_time, motor_speed=MOTOR_SPEED):
    pinky_motor.move(motor_speed + MOTOR_TRIM, motor_speed - MOTOR_TRIM)
    time.sleep(duration_time)
    pinky_motor.move(0, 0)

def move_backward(duration_time, motor_speed=MOTOR_SPEED):
    # 후진은 같은 편향이 반대 방향으로 나타난다 -> 부호도 뒤집는다
    pinky_motor.move(-motor_speed + MOTOR_TRIM, -motor_speed - MOTOR_TRIM)
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

    time.sleep(TURN_SETTLE_BASE + TURN_SETTLE_PER_SPEED * speed)  # coast 끝날 때까지
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
# 차선(도로 마스크) 추종을 쓸지. False 면 그냥 직진한다.
# 코스가 고정이고 구간이 직선이며 교차로마다 마커로 위치를 리셋하므로, 카메라를
# 0.25초마다 보는 비용을 안 내고 그냥 달리는 편이 빠르다. 대신 도로 이탈(5초/회)을
# 막아줄 게 없으니, 굽은 구간이 있으면 True 로 되돌린다.
ROAD_FOLLOW = True

# 마커까지 외워둔 거리 중 몇 %를 카메라 없이 먼저 달릴지. 나머지는 마커로 폐루프.
# 1.0 에 가까울수록 빠르지만 마커를 지나칠 위험이 커진다.
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
# 0.4 였는데 0.15 로 내렸다. 0.4초는 50cm/s 에서 20cm 이고, 도로 폭이 15.4cm 인데
# 좌우 여유가 2.1cm 뿐이라 20cm 를 눈 감고 가면 곡선에서 그대로 나간다.

# 카메라 광축이 로봇 중심선과 어긋난 양(px, 오른쪽이 +). 0 이 아닌데 0 으로 두면
# 로봇은 "화면 중앙 = 도로 중앙" 으로 착각해 그 차이만큼 계속 한쪽에 붙어 달린다.
# 좌우 여유가 2.1cm 뿐이라 이 편향 하나로 바퀴가 선을 넘는다.
# 도로 한가운데에 똑바로 세우고 `python3 drive.py road` -> 추천값을 여기 적는다.
ROAD_CENTER_BIAS = 0



def _otsu(ch, lo, hi):
    """이 프레임 안에서 밝은/어두운(또는 저채도/고채도) 경계를 스스로 찾는다.

    조명이 바뀌면 고정 임계값이 통째로 깨지는데, 대회 당일 조명은 미리 알 수 없고
    현장에서 튜닝할 시간도 보장되지 않는다. Otsu 는 히스토그램 골짜기를 찾으므로
    조명이 밝아지면 임계값도 같이 올라간다. 다만 화면에 도로만(또는 잔디만) 있으면
    엉뚱한 데를 자르므로, 실측값 주변으로 clamp 해서 폭주를 막는다.
    """
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


def road_band_width(mask, band):
    """밴드에서 가장 넓은 연속 도로 구간의 폭(px). 교차로 감지에 쓴다.

    교차로에서는 옆으로 갈라진 길이 같은 밴드에 붙어 나타나므로 이 폭이 급격히
    넓어진다. 그 정점이 "지금 교차로 한가운데" 다.
    """
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
    error = (w / 2 + ROAD_CENTER_BIAS - near) / (w / 2)
    curve = (near - far) / (w / 2)
    return float(np.clip(error, -1, 1)), float(np.clip(curve, -1, 1))


def road_steer(frame):
    """조향량 -1(좌) ~ +1(우). 도로를 못 찾으면 None -> 직진 유지."""
    error, curve = road_offset(frame)
    if error is None:
        return None
    return float(np.clip(-(ROAD_KP * error + ROAD_KD * curve), -1, 1))



# ================================================================ 외운 경로 주행

# 대회 도면(26wcrc_final.pdf 3페이지)에서 딴 도로 중심선. (누적 cm, 상대 헤딩 도).
#
# 카메라 도로 추종은 이 코스에서 못 쓴다 — S자라 직진 구간이 없어서 근거리 도로가
# 항상 화면 밖으로 잘리고, 그러면 "가장 넓은 구간의 중점" 이 도로 중심이 아니게 된다.
# error 가 실제 치우침보다 작게 나와 로봇은 한쪽에 붙은 채 평형을 이룬다 (실측).
# 그래서 도로 형상을 미리 외워 IMU 로 닫는다.
#
# 헤딩은 구간 시작을 0 으로 본 **상대값**이다. 교차로에서 마커 정렬 회전이 끼어도
# 다음 호출이 그때의 헤딩을 새 기준으로 잡으므로 저절로 흡수된다.
# 도면은 200dpi 렌더에서 1px = 1mm 로 떨어진다 (경기장 2100x1100mm).
# 5cm 웨이포인트 선형보간의 실제 중심선 대비 최대 이탈 1.0cm (좌우 여유 2.1cm).
# 등곡률 원호로 근사하면 마지막 구간에서 12.3cm 벗어나 도로를 나간다 — 그래서 안 쓴다.
PATH_LEGS = [
    # [0] START -> 교차로1 (마커1)  도면 48cm, 총 +13도
    [(0, +0), (9, -3), (13, -6), (20, -3), (23, +1), (25, +4), (27, +7), (29, +10), (33, +13), (48, +13)],
    # [1] 교차로1 -> 교차로2 (마커2)  도면 48cm, 총 +6도
    [(0, +0), (14, +3), (18, +7), (22, +10), (34, +6), (48, +6)],
    # [2] 교차로2 -> 교차로3 (마커4)  도면 47cm, 총 +24도
    [(0, +0), (26, +3), (28, +6), (29, +9), (31, +12), (33, +16), (34, +19), (34, +22), (47, +24)],
    # [3] 교차로3 -> 교차로4 (마커5)  도면 51cm, 총 -6도
    [(0, +0), (20, +3), (27, +0), (32, -3), (38, -6), (51, -6)],
    # [4] 교차로4 -> 정지선 -> END (마커10)  도면 67cm, 총 -76도
    [(0, +0), (14, -3), (18, -7), (19, -10), (19, -13), (20, -16), (21, -19), (22, -22), (22, -26), (23, -29), (27, -32), (32, -29), (33, -25), (33, -22), (34, -19), (35, -16), (37, -13), (43, -17), (43, -20), (44, -24), (44, -27), (45, -30), (45, -34), (46, -37), (46, -40), (47, -43), (48, -47), (49, -50), (49, -53), (50, -56), (50, -60), (51, -63), (51, -66), (51, -70), (52, -73), (52, -76), (67, -76)],
]

PATH_MODE = False       # run2 가 켠다. run 은 건드리지 않는다
PATH_STEP_CM = 3.0      # 한 번에 눈 감고 가는 거리. 짧을수록 정확하고 느리다
PATH_KP = 0.5           # 한 스텝에 헤딩 오차의 몇 배를 없앨지. 1.0 이면 한 번에 다 (진동)
PATH_MAX_BIAS = 35      # 좌우 속도차 상한. 넘으면 속도를 낮춘다 (아래 참고)
PATH_MIN_SPEED = 35     # 급커브에서 낮출 수 있는 속도의 바닥

_leg = None             # 지금 타고 있는 구간 프로파일
_leg_cm = 0.0           # 그 구간에서 지금까지 간 거리


def path_heading(cm):
    """구간 시작에서 cm 만큼 갔을 때의 목표 헤딩(도). 프로파일 밖이면 끝값을 문다."""
    if not _leg:
        return 0.0
    return float(np.interp(cm, [p[0] for p in _leg], [p[1] for p in _leg]))


def follow_path(duration_time, motor_speed=MOTOR_SPEED):
    """외운 헤딩 프로파일을 따라 전진한다. duration_time 은 기준 속도 기준 시간 = 곧 거리.

    좌우 속도차는 두 몫을 더한 것이다.
      · 피드포워드: 이 스텝에서 돌아야 할 각도를 그대로 만든다. 회전 실측식을 쓴다 —
        좌우차 D 가 만드는 각속도가 TURN_DEG_PER_SEC x (D/2) / MOTOR_SPEED 이므로
        원하는 각속도에서 D 를 역산할 수 있다. MOTOR_TRIM 과 같은 환산이다.
      · 피드백: IMU 로 잰 '아직 못 돈 각도'. 미끄러짐과 상수 편향을 여기서 턴다.
        IMU 를 못 읽으면 피드포워드만으로 간다 (열린 루프).
    """
    global _leg_cm
    dist = duration_time * MOVE_FORWARD_PER_ONE
    y0 = read_yaw()
    h0 = path_heading(_leg_cm)
    gone = 0.0
    while gone < dist - 1e-6:
        step = min(PATH_STEP_CM, dist - gone)
        sec = step / MOVE_FORWARD_PER_ONE
        turn = path_heading(_leg_cm + step) - path_heading(_leg_cm)
        err = 0.0
        if y0 is not None:
            y = read_yaw()
            if y is not None:
                err = (path_heading(_leg_cm) - h0) - ((y - y0 + 180) % 360 - 180)
        bias = MOTOR_SPEED * ((turn + PATH_KP * err) / sec) / TURN_DEG_PER_SEC
        speed = motor_speed
        if abs(bias) > PATH_MAX_BIAS:
            # 급커브. 좌우차만 키우면 한쪽 바퀴가 상한에 잘려 오히려 덜 휜다.
            # 곡률은 (좌우차 / 속도) 에 비례하므로 속도를 낮추면 같은 차이로 더 휜다.
            # 시간은 그만큼 길어지고 가는 거리는 그대로다 (거리로 진행을 센다).
            scale = max(PATH_MIN_SPEED / speed, PATH_MAX_BIAS / abs(bias))
            speed = max(PATH_MIN_SPEED, int(speed * scale))
            bias *= scale
            sec = step / (MOVE_FORWARD_PER_ONE * speed / MOTOR_SPEED)
        bias = int(np.clip(bias, -PATH_MAX_BIAS, PATH_MAX_BIAS))
        pinky_motor.move(max(10, min(100, speed + bias + MOTOR_TRIM)),
                         max(10, min(100, speed - bias - MOTOR_TRIM)))
        time.sleep(sec)
        gone += step
        _leg_cm += step
    pinky_motor.move(0, 0)      # 구간 끝에서만 선다


def leg_begin(i):
    """i 번째 구간의 외운 경로를 건다. run(PATH_MODE=False) 이면 아무것도 안 한다."""
    global _leg, _leg_cm
    if PATH_MODE and i < len(PATH_LEGS):
        _leg, _leg_cm = PATH_LEGS[i], 0.0
        print(f"  외운 경로 [{i}] 총 {_leg[-1][0]:.0f}cm / {_leg[-1][1]:+.0f}도")

def move_forward_on_road(duration_time, step=0.12, motor_speed=MOTOR_SPEED):
    """step 은 0.25 였는데 0.12 로 내렸다.

    바퀴 폭 111mm / 흰 도로 154mm → 좌우 여유 21.5mm. 곡선 반경이 약 55cm 이므로
    한 스텝에 d 만큼 가면 도로가 d^2/(2*55cm) 만큼 옆으로 빠진다.
    0.25초(12.5cm) 면 1.4cm — 여유 2.1cm 를 거의 다 먹는다. 0.12초(6cm) 면 0.3cm.
    """
    """도로 중심을 보며 전진한다.

    긴 전진을 짧게 쪼개고, 매 조각 직전에 도로를 보고 좌우 바퀴 속도를 다르게 준다.
    원본은 목표까지 계산한 시간만큼 눈 감고 직진해서, 도로가 휘면 그대로 잔디로 나갔다.

    짧은 전진(마지막 미세 접근)은 조향하지 않는다. 이미 아루코로 정렬된 상태라
    거기서 또 꺾으면 정렬이 흐트러진다.
    """
    if duration_time < ROAD_MIN_STEER_TIME:
        globals()["_leg_cm"] += duration_time * MOVE_FORWARD_PER_ONE
        move_forward(duration_time, motor_speed)
        return

    if _leg is not None:
        # run2. 카메라 대신 외운 경로를 탄다 (ROAD_FOLLOW 와 무관하다).
        return follow_path(duration_time, motor_speed)

    if not ROAD_FOLLOW:
        return move_forward(duration_time, motor_speed)

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
    output_frame, pose = pinky_cam.detect_aruco(frame, marker_size=MARKER_SIZE_M)

    if not pose:
        print("None is detected")
        return False, None

    for p in pose:
        if int(p[0]) != aruco_num:
            continue
        z = p[3]
        # 거리로 한 번 거른다. 36mm 마커는 실측상 37~75cm 에서만 제대로 읽힌다
        # (그 밖은 10프레임 중 0회). 그런데도 "보인다" 고 나오면 십중팔구
        # 멀리 있는 다른 마커를 잘못 읽은 것이고, 그걸 목표로 삼으면 엉뚱한 데로 간다.
        if not (Z_ACCEPT_MIN <= z <= Z_ACCEPT_MAX):
            print(f"Target id:{aruco_num} z:{z:.1f} — 거리 밖이라 무시 "
                  f"({Z_ACCEPT_MIN}~{Z_ACCEPT_MAX}cm 만 인정)")
            continue
        print(f"Target detected  id:{int(p[0])} x:{p[1]:.1f} y:{p[2]:.1f} z:{z:.1f}")
        return True, [p]              # 호출부가 pose[0][n] 을 쓰므로 리스트로 감싼다

    print("Target not detected (보이는 id:", [int(p[0]) for p in pose], ")")
    return False, None


# direction : 0(Left), 1(Right)
def find_aruco_with_try_count(aruco_num, direction, try_count, deadline=None):
    """한 방향으로 훑으며 마커를 찾는다. 찾는 즉시 멈춘다.

    예전엔 스텝마다 IMU 폐루프 회전(turn_deg)을 썼다. 그건 coast 대기 0.34초가
    붙어 있어서 한 스텝에 1.4초씩 걸렸고, 그래서 탐색이 뚝뚝 끊겼다.
    탐색에는 1도 정밀도가 필요 없다 — 마커가 화면에 들어오기만 하면 되고, 그 다음
    각도 정렬 루프가 알아서 맞춘다. 그래서 여기서는 대기 없는 짧은 회전을 이어 붙인다.

    대신 "얼마나 돌았는지" 는 IMU 로 직접 재서, 못 찾았을 때 정확히 되돌아온다.
    (열린 회전을 명령각만 믿고 되돌리면 슬립만큼 방향이 틀어진 채로 남는다)
    """
    if try_count == 0:
        return False, None

    span = SEARCH_STEP_DEG * try_count
    rate = TURN_DEG_PER_SEC * SEARCH_SCAN_SPEED / MOTOR_SPEED   # 도/초
    chunk = SEARCH_STEP_DEG / rate                              # 한 스텝 회전 시간
    turn = move_left if direction == LEFT else move_right

    y0 = read_yaw()
    for i in range(try_count):
        if deadline and time.time() > deadline:
            print("  탐색 중단 (시간 초과)")
            break
        success, pose = detect_target_aruco(aruco_num)
        if success:
            print("find_aruco, detected")
            return True, pose
        turn(chunk, SEARCH_SCAN_SPEED)
        print(f"search {i + 1}/{try_count} ({SEARCH_STEP_DEG * (i + 1)}/{span}도)")

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
    """지정 방향 -> 반대 방향으로 훑고, 그래도 없으면 조금 전진해서 다시 본다.

    제자리 회전만으로는 "아직 너무 멀어서 안 보이는" 경우를 못 푼다. 36mm 마커는
    75cm 밖이면 픽셀이 모자라 아예 안 잡히므로, 그때는 각도가 아니라 거리가 문제다.
    코스가 고정이라 앞으로 가는 건 안전하다 — 어차피 그 방향으로 갈 참이다.
    """
    for attempt in range(FIND_ADVANCE_TRIES + 1):
        if deadline and time.time() > deadline:
            print(f"  마커 {aruco_num} 탐색 시간 초과")
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
            print(f"  마커 {aruco_num} 안 보임 — {FIND_ADVANCE_CM}cm 전진 후 재탐색 "
                  f"({attempt + 1}/{FIND_ADVANCE_TRIES})")
            move_forward_on_road(FIND_ADVANCE_CM / MOVE_FORWARD_PER_ONE)
            time.sleep(SLEEP_TIME_AFTER_MOVE)
    return False, None

def check_angle(aruco_num, target, allow_range=15):
    """(맞았나, 돌아야 할 각도) 를 돌려준다. 못 보면 (False, None).

    예전엔 방향(LEFT/RIGHT)만 주고 호출부가 MATCH_STEP_DEG 씩 찔끔찔끔 돌았다.
    20도 어긋나 있으면 네 번을 돌아야 하고, 회전 한 번마다 coast 대기가 붙는다.
    x(좌우 cm)와 z(거리 cm)를 둘 다 아는데 각도를 모를 리가 없다 — 바로 계산한다.
    """
    success, pose = detect_target_aruco(aruco_num)
    if not success:
        print("check_angle, not detected")
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
    """(도착했나, 전진할 시간, 지금 본 z) 를 돌려준다.

    z 를 같이 돌려주는 이유: 예전엔 호출부가 실측 로그를 찍으려고 detect_target_aruco
    를 한 번 더 불렀다. 접근 루프가 반복문이라 프레임을 매번 두 장씩 읽었고,
    그게 그대로 랩타임이었다.
    """
    global _last_seen_x
    success, pose = detect_target_aruco(aruco_num)
    if not success:
        print("check_distance, not detected")
        return False, None, None

    _last_seen_x = pose[0][1]
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
    """1) 마커 탐색 -> 2) 각도(x) 정렬 -> 3) 거리(z) 접근.

    원본 대비 바뀐 점 두 가지 (둘 다 안 고치면 대회에서 그대로 물린다):
      - 각도 정렬 루프에 탈출구가 없었다. check_angle 이 마커를 놓치면 (False, None) 을
        주는데 받는 쪽이 `if LEFT ... else move_right` 라서 None 이 전부 우회전으로 떨어져
        영원히 제자리 우회전을 했다. → 규정 패널티 1번(60초 정체 시 기회 종료) 직행.
      - 마커 하나에 무한정 매달렸다. → timeout 초를 넘기면 포기하고 False 를 돌려준다.
    """
    target_x, target_z, target_direction = target_pose
    deadline = time.time() + timeout

    ok, pose = find_aruco(aruco_num, target_direction, try_count, deadline)
    if not ok:
        print("track_target_aruco_marker find_aruco_failed")
        return False

    # --- 각도(x) 맞추기 ---
    lost = 0
    while True:
        if time.time() > deadline:
            print(f"[timeout] 마커 {aruco_num} 각도 정렬 {timeout}초 초과, 포기")
            return False

        aligned, angle_deg = check_angle(aruco_num, target_x)
        if aligned:
            break

        if angle_deg is None:                # 마커를 놓쳤다
            lost += 1
            print("angle: 마커 놓침", lost)
            if lost >= 5:
                lost = 0
                ok, _ = find_aruco(aruco_num, target_direction, try_count, deadline)
                if not ok:
                    print("angle: 재탐색 실패")
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
    global _last_arrival_z, _last_arrival_x, _last_seen_x
    _last_arrival_z = _last_arrival_x = _last_seen_x = None
    not_detected_count = 0
    last_z, last_step = None, None
    while True:
        if time.time() > deadline:
            print(f"[timeout] 마커 {aruco_num} 거리 접근 {timeout}초 초과, 여기서 멈춤")
            break

        arrived, distance_direction, z_now = check_distance(aruco_num, target_z)

        # 직전 스텝이 실제로 몇 cm 를 갔는지 관찰한다. 이 값이 상수와 크게 다르면
        # forward-cal 로 MOVE_FORWARD_PER_ONE 을 다시 재야 한다는 뜻이다.
        if z_now is not None:
            if last_z is not None and last_step:
                moved = last_z - z_now
                print(f"    실측 전진 {moved:.0f}cm / {last_step:.2f}초 "
                      f"= {moved / last_step:.0f}cm/s (상수 {MOVE_FORWARD_PER_ONE:.0f})")
            last_z = z_now

        if arrived:
            _last_arrival_z = last_z
            _last_arrival_x = _last_seen_x
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
            total_apple_count += collect_apple_counts()
            display_apple_count(total_apple_count)
        elif action_inside == APPLE_COUNT_ACTION:
            start_apple_count()
            print("사진 3장 촬영 — 서버 전송은 주행하면서 백그라운드로")
        elif action_inside == GO_TO_MARKER:
            # 마커를 잡았을 때 실제로 잰 거리를 쓴다. 고정 시간보다 정확하다.
            # z 는 "마커 고정 시점" 기준이므로, 그 뒤에 이미 간 hop 만큼은 빼야 한다.
            # (안 빼면 hop 을 조정할 때마다 END 위치가 같이 밀린다)
            z = _last_arrival_z
            if z is None:
                print("  마커 거리를 모른다 — 전진 생략")
            else:
                extra = 0 if option_inside == DEFAULT else option_inside
                done = target_list[index].get("hop") or 0
                remain = z - done + extra
                print(f"  마커까지 잰 {z:.0f}cm - 이미 간 {done}cm + {extra}cm "
                      f"= {remain:.0f}cm 전진")
                if remain > 0:
                    move_forward_on_road(remain / MOVE_FORWARD_PER_ONE)
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

    os.makedirs(APPLE_SHOT_DIR, exist_ok=True)
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
    global total_apple_count
    _cal.clear()
    assert len(target_list) == len(after_track_list), \
        "target_list 와 after_track_list 의 개수가 다릅니다"
    for t in target_list:
        x, _, d = t["pose"]
        # 마커가 왼쪽에 있는데 오른쪽부터 훑으면 120도를 헛돈다. 손으로 맞추면
        # 언젠가 어긋나므로 출발 전에 막는다.
        assert x == 0 or (x < 0) == (d == LEFT), \
            f"마커 {t['id']}: x={x} 인데 탐색 방향이 반대다 (x<0 이면 LEFT)"

    start_time = time.time()
    for i in range(len(target_list)):
        current_id = target_list[i]["id"]
        leg_begin(i)
        print(f"\n===== [{i}] 마커 {current_id} 로 이동 "
              f"(경과 {time.time() - start_time:.0f}s) =====")

        # 맵을 외웠으니 대부분의 거리는 카메라를 안 보고 먼저 간다. 마커는 마지막
        # 구간에서 "정확히 어디서 서고 어느 쪽을 보고 있나" 를 잡는 용도로만 쓴다.
        cm = target_list[i].get("cm")
        if cm:
            blind = cm * APPROACH_BLIND_RATIO
            print(f"  외운 거리 {cm}cm 중 {blind:.0f}cm 를 먼저 이동")
            move_forward_on_road(blind / MOVE_FORWARD_PER_ONE)

        result = track_target_aruco_marker(current_id, target_list[i]["pose"], SEARCH_COUNT)
        _cal.append((current_id, cm, target_list[i]["pose"][1], _last_arrival_z,
                     _last_arrival_x, result))
        if not result:
            # 평가는 "어디까지 갔나" 로 그룹이 갈린다 (메인도로 E < 교차로1 D < 과수원 C
            # < 도착 B < 하차장 A). 마커 하나 놓쳤다고 전체를 포기하면 그룹이 내려간다.
            #
            # 그리고 그냥 넘어가면 안 된다. 못 찾았다는 건 아직 "직전 지점"에 서 있다는
            # 뜻이라, 여기서 복귀 동작을 해봐야 엉뚱한 자리에서 도는 것이고 다음 마커는
            # 더 못 찾는다. 코스가 고정이니 외워둔 거리만큼 도로를 따라 밀고 간다.
            print(f"!! 마커 {current_id} 실패 — 남은 거리만 마저 가고 진행")
            if cm:
                move_forward_on_road(cm * (1 - APPROACH_BLIND_RATIO) / MOVE_FORWARD_PER_ONE)

        # 추적에 실패했어도 로봇은 이미 그 자리에 있다. 복귀 동작을 안 하면 갇히므로
        # 성공/실패와 무관하게 실행한다.
        # 마커로 위치를 확정했으면, 교차로까지 남은 거리는 눈 감고 간다.
        # (실패했을 때는 위에서 이미 남은 거리를 갔으므로 두 번 가지 않는다)
        if result and cm:
            # 마커를 잡았다 = 지금이 외운 cm 지점이다. 정렬·접근하며 흘린 거리를 여기서
            # 맞춰야 남은 구간의 곡률 타이밍이 안 밀린다.
            globals()["_leg_cm"] = float(cm)

        hop = target_list[i].get("hop") or 0
        if result and hop:
            print(f"  마커 확정 -> 교차로까지 외운 {hop}cm 를 마저 간다")
            move_forward_on_road(hop / MOVE_FORWARD_PER_ONE)

        after_target_do_list(i)
        print(f"----list num : {i} done -----")

    pinky_motor.move(0, 0)
    print("\n===== 거리 캘리브레이션 =====")
    print(" 마커   cm   목표z   실제z=권장hop   옆거리   판정")
    for mid, cm, tz, az, ax, ok in _cal:
        if az is None:
            print(f" {mid:>4}  {cm:>4}   {tz:>4}    ----     ----   마커 실패 — cm 이 너무 길거나 짧다")
            continue
        # 마커가 교차로 옆에 붙어 있으므로 "마커까지 남은 깊이" 가 곧 교차로까지의 거리다.
        # (마커1 에서 z=21.2 / hop 23 으로 완주 성공한 것이 이 규칙의 근거)
        hop_now = next(t["hop"] for t in target_list if t["id"] == mid)
        if mid == 10:
            # 마커10 의 hop 은 "교차로까지" 가 아니라 "횡단보도 정지선까지" 라 규칙이 다르다.
            # END 까지는 이 z 를 그대로 써서 가므로(GO_TO_MARKER) 손댈 게 없다.
            note = "정지선 기준 — hop 규칙 해당 없음"
        else:
            note = "OK" if abs(az - hop_now) <= 4 else f"hop {hop_now} -> {az:.0f} 로 바꿀 것"
        # 회전은 마커와 나란한 자리에서 한다. 옆거리가 회전 반경보다 좁으면 몸체가
        # 마커를 친다 = 규정 5번 5초. 도로 중앙을 못 지키면 여기서 여유가 먼저 사라진다.
        if ax is None:
            side = " ----"
        else:
            gap = abs(ax) - ROBOT_SWEEP_CM
            side = f"{abs(ax):5.1f}"
            if gap < 0:
                note = f"!! 마커를 친다 (여유 {gap:+.1f}cm) — " + note
            elif gap < 2:
                note = f"!  아슬아슬 (여유 {gap:+.1f}cm) — " + note
        print(f" {mid:>4}  {cm:>4}   {tz:>4}   {az:5.1f}   {side}   {note}")
    total_apple_count += collect_apple_counts()   # 아직 안 거둔 게 있으면 여기서
    print(f"\n===== 주행 종료. 사과 {total_apple_count}개 / "
          f"{time.time() - start_time:.0f}초 =====")
    display_apple_count(total_apple_count)


# ================================================================ 서브커맨드
def warmup_motors():
    """출발 전에 모터를 예열한다.

    차가운 모터는 첫 이동이 26% 느리다 (실측 13.2 -> 17.7 cm/s). 첫 구간이 29cm 라
    그대로 두면 7.6cm 모자란 채로 마커를 찾게 된다.
    출발선 위치를 바꾸면 안 되므로 제자리 회전만 쓴다. IMU 로 닫혀 있어 헤딩이
    처음으로 돌아온다.
    """
    if pinky_motor is None:
        return
    print("모터 예열 (제자리, 출발 위치 안 바뀜)")
    for deg in (10, -10, 10, -10):
        turn_deg(deg)


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
        warmup_motors()
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


def cmd_run2(*args):
    """run 과 같은 코스를, 카메라 도로 추종 대신 **외운 경로**로 달린다.

        python3 drive.py run2

    도로 형상은 대회 도면에서 딴 PATH_LEGS 이고, 헤딩은 IMU 로 닫는다.
    마커 탐색·정렬·교차로 동작은 run 과 완전히 같은 코드를 쓴다 — 다른 건 조향뿐이다.
    run 은 그대로 두었으니, 이쪽이 어긋나면 run 으로 돌아가면 된다.
    """
    global PATH_MODE
    PATH_MODE = True
    print("외운 경로 모드 (카메라 도로 추종 끔)")
    return cmd_run(*args)


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


# ================================================================ 캘리브레이션


def _write_const(name, value, path):
    """`NAME = 숫자` 한 줄의 값만 바꾼다. 못 찾으면 예외 — 조용히 넘어가면
    캘리브레이션을 돌리고도 옛날 값으로 달리게 된다."""
    src = open(path, encoding="utf-8").read()
    pat = re.compile(rf"^({name} = )-?[\d.]+", re.M)
    if not pat.search(src):
        raise KeyError(f"{name} 을 {path} 에서 못 찾았다")
    open(path, "w", encoding="utf-8").write(pat.sub(rf"\g<1>{value}", src, count=1))
    return next(l for l in open(path, encoding="utf-8") if l.startswith(name + " ")).rstrip()


def _spin_measure(speed, deg):
    """speed 로 deg 도 돌려 (각속도 도/초, coast 도) 를 잰다. 양수 = 우회전.

    turn_deg 를 쓰지 않는다 — 그건 지금 재려는 상수로 미리 멈추기 때문에
    그걸로 재면 옛 값이 그대로 나오는 순환이 된다. 여기선 목표각까지 돌린 뒤
    끄고, 꺼진 다음 더 돈 각도를 coast 로 본다.
    """
    y = read_yaw()
    if y is None:
        return None, None
    sign = 1 if deg > 0 else -1
    prev, turned = y, 0.0
    pinky_motor.move(speed * sign, -speed * sign)
    t0 = time.time()
    while abs(turned) < abs(deg) and time.time() - t0 < TURN_MAX_SEC:
        y = read_yaw()
        if y is not None:
            turned += (y - prev + 180) % 360 - 180
            prev = y
    pinky_motor.move(0, 0)
    dt = time.time() - t0
    time.sleep(0.9)                       # coast 가 완전히 멎을 때까지
    y = read_yaw()
    coast = abs((y - prev + 180) % 360 - 180) if y is not None else 0.0
    return abs(turned) / dt, coast


def _marker_z(tries=9, need=3):
    """지금 보이는 아루코까지의 z(cm) 의 **중앙값**. 못 보면 None.

    한 프레임만 쓰면 못 쓴다. 45cm 에서 36mm 마커의 z 는 프레임마다 몇 cm 씩 튀고,
    10cm 이동을 z 차이로 재면 그 노이즈가 그대로 60% 오차가 된다 (실측).
    중앙값이라 한두 장이 크게 튀어도 견딘다.
    """
    zs = []
    for _ in range(tries):
        _, p = pinky_cam.detect_aruco(pinky_cam.get_frame(), marker_size=MARKER_SIZE_M)
        if p:
            zs.append(p[0][3])
    return float(np.median(zs)) if len(zs) >= need else None


CAL_TURN_SPEEDS = (15, 25, 40, 60, MOTOR_SPEED)
CAL_TURN_DEG = 60          # 한 번에 도는 각도. 좌우 한 쌍이라 제자리로 돌아온다
# 길게 갈수록 z 노이즈에 덜 휘둘린다. 다만 너무 가면 마커 인식 범위(37~75cm)를
# 벗어난다 — 60cm 에서 시작하면 1.2초(약 20cm)가 딱 맞는다.
CAL_FORWARD_SEC = 1.2
CAL_FORWARD_N = 5


def cmd_cal(*args):
    """주행 상수를 한 번에 다시 잰다. 무게가 바뀌면(센서 탈착·배터리 교체) 이것만 돌린다.

        python3 drive.py cal                 # 전부 재고, 확인 후 파일에 기록
        python3 drive.py cal turn            # 회전만
        python3 drive.py cal forward road    # 골라서
        python3 drive.py cal --yes           # 묻지 않고 바로 기록

    준비 (한 자리에서 셋 다 된다):
      · 로봇을 도로 한가운데, 진행 방향으로 똑바로   -> ROAD_CENTER_BIAS
      · 정면 45~65cm 에 아루코 마커 하나            -> MOVE_FORWARD_PER_ONE
      · 좌우로 한 바퀴 돌 공간                      -> TURN_DEG_PER_SEC, TURN_COAST_PER_SPEED
    끝나면 로봇은 처음 자리·처음 방향으로 돌아와 있다.
    """
    args = list(args)
    auto = "--yes" in args
    if auto:
        args.remove("--yes")
    only = set(args) or {"road", "turn", "forward"}
    unknown = only - {"road", "turn", "forward"}
    assert not unknown, f"모르는 항목: {unknown}"

    new = {}
    setup()
    try:
        if "road" in only:
            frame = pinky_cam.get_frame()
            e, _ = road_offset(frame)
            if e is None:
                print("[도로] 도로가 안 보인다 -> 건너뜀")
            else:
                new["ROAD_CENTER_BIAS"] = round(ROAD_CENTER_BIAS - e * frame.shape[1] / 2)
                print(f"[도로] error={e:+.3f} -> ROAD_CENTER_BIAS = {new['ROAD_CENTER_BIAS']}")

        if only & {"turn", "forward"}:
            warmup_motors()      # 냉간은 26% 느리다. 예열 전 값을 적으면 코스가 통째로 밀린다

        if "turn" in only:
            if read_yaw() is None:
                print("[회전] IMU 를 못 읽는다 -> 건너뜀")
            else:
                rates, coasts = [], []
                for speed in CAL_TURN_SPEEDS:
                    for sign in (1, -1):
                        rate, coast = _spin_measure(speed, sign * CAL_TURN_DEG)
                        if rate is None:
                            continue
                        rates.append(rate / speed)
                        coasts.append((speed, coast))
                        print(f"  속도 {speed:>2} {'우' if sign > 0 else '좌'}: "
                              f"{rate:5.0f}도/초   coast {coast:4.1f}도")
                if coasts:
                    new["TURN_DEG_PER_SEC"] = round(float(np.median(rates)) * MOTOR_SPEED)
                    # coast = k x 속도 (원점을 지나는 직선) 의 최소제곱해
                    v = np.array([s for s, _ in coasts], float)
                    c = np.array([x for _, x in coasts], float)
                    new["TURN_COAST_PER_SPEED"] = round(float((v * c).sum() / (v * v).sum()), 3)
                    print(f"[회전] TURN_DEG_PER_SEC = {new['TURN_DEG_PER_SEC']}   "
                          f"TURN_COAST_PER_SPEED = {new['TURN_COAST_PER_SPEED']}")

        if "forward" in only:
            speeds, drifts = [], []
            for i in range(CAL_FORWARD_N):
                # 뒤로 먼저 뺀다. 마커 인식은 37~75cm 에서만 되므로, 앞으로만 가면
                # 잴 수 있는 거리가 얼마 안 된다. 뒤로 뺐다가 그 거리를 재면
                # 마커를 옮기지 않고도 길게 잴 수 있다 (긴 거리 = 작은 상대 오차).
                move_backward(CAL_FORWARD_SEC)
                time.sleep(0.4)                       # 흔들림이 멎어야 z 가 튀지 않는다
                z0, y0 = _marker_z(), read_yaw()
                move_forward(CAL_FORWARD_SEC)
                time.sleep(0.4)
                z1, y1 = _marker_z(), read_yaw()
                if y0 is not None and y1 is not None:
                    # 똑바로 갔다면 yaw 가 그대로여야 한다. 돈 만큼이 좌우 편향이다.
                    drifts.append(((y1 - y0 + 180) % 360 - 180) / CAL_FORWARD_SEC)
                if z0 is None or z1 is None:
                    print(f"  {i + 1}회차: 마커를 놓쳤다 — 속도는 버린다"
                          f"{'' if not drifts else f'  (휨 {drifts[-1]:+.1f}도/초)'}")
                    continue
                speeds.append(abs(z1 - z0) / CAL_FORWARD_SEC)
                print(f"  {i + 1}회차: z {z0:5.1f} -> {z1:5.1f}cm   {speeds[-1]:5.2f} cm/초"
                      f"{'' if not drifts else f'   휨 {drifts[-1]:+.1f}도/초'}")
            if speeds:
                med = float(np.median(speeds))
                spread = (max(speeds) - min(speeds)) / med
                if spread > 0.25 or len(speeds) < 3:
                    # 실제 이동 속도가 회차마다 25% 씩 다를 리 없다. z 가 튄 것이므로
                    # 이 값을 적으면 외운 거리가 통째로 틀어진다. 차라리 안 적는다.
                    print(f"[전진] 편차 {spread*100:.0f}% ({len(speeds)}회) — 못 믿는다, 적지 않는다."
                          f"  마커를 60cm 앞에 두고 다시")
                else:
                    new["MOVE_FORWARD_PER_ONE"] = round(med, 2)
                    print(f"[전진] MOVE_FORWARD_PER_ONE = {new['MOVE_FORWARD_PER_ONE']}"
                          f"  (편차 {spread*100:.0f}%)")
            else:
                print("[전진] 전부 실패 -> 마커를 45~65cm 정면에 두고 다시")
            if drifts:
                # 좌우 차 D 가 만드는 각속도는 TURN_DEG_PER_SEC x (D/2) / MOTOR_SPEED.
                # trim 은 한쪽에 더하고 반대쪽에서 빼므로 D = 2 x trim 이다.
                drift = float(np.median(drifts))
                new["MOTOR_TRIM"] = round(MOTOR_TRIM - drift * MOTOR_SPEED / TURN_DEG_PER_SEC)
                print(f"[직진] 휨 {drift:+.1f}도/초 -> MOTOR_TRIM = {new['MOTOR_TRIM']}")
                if abs(new["MOTOR_TRIM"] - MOTOR_TRIM) > 1:
                    print("       한 번에 다 안 잡힌다 (직진과 회전은 미끄러짐이 다르다)."
                          " 기록하고 cal forward 를 한 번 더 돌릴 것")
            else:
                print("[직진] IMU 를 못 읽는다 -> MOTOR_TRIM 은 못 잰다")
    finally:
        teardown()

    if not new:
        print("\n잰 값이 없다 — 기록하지 않는다")
        return 1
    # 여기서 재는 건 '제자리 실험' 값이고, 지금 파일에 있는 건 코스를 실제로 돌려
    # 맞춘 값일 수 있다. 실주행 튜닝을 실험값으로 덮으면 주행이 나빠진다 — 실제로
    # 그렇게 해서 한 번 망쳤다. 크게 다르면 눈에 띄게 세우고, --yes 여도 묻는다.
    big = {k: v for k, v in new.items()
           if globals()[k] and abs(v - globals()[k]) / abs(globals()[k]) > 0.05}
    print("\n적용할 값:")
    for k, v in new.items():
        cur = globals()[k]
        mark = "  <<< 5% 넘게 바뀐다. 지금 값이 실주행으로 맞춘 것이면 두는 게 낫다" if k in big else ""
        print(f"  {k}: {cur} -> {v}{mark}")
    if big and auto:
        print("\n실주행 튜닝을 덮을 수 있어 --yes 를 무시한다. 확인이 필요하다.")
        auto = False
    if not auto and input("엔터 -> 기록 / 그 외 입력 -> 취소: ").strip():
        print("취소")
        return 0

    path = os.path.abspath(__file__)
    shutil.copy(path, path + ".bak")
    print(f"\n백업 {path}.bak")
    for k, v in new.items():
        print("  " + _write_const(k, v, path))
    print("주석은 손대지 않는다 — 위 줄을 보고 옛 설명이 남아 있으면 직접 고칠 것")
    print("바뀐 값은 다음 실행부터 적용된다 (지금 프로세스는 옛 값)")
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
    if e is not None:
        # 지금 도로 한가운데에 똑바로 서 있다면, 남은 error 는 전부 카메라 편향이다.
        print(f"추천  ROAD_CENTER_BIAS = {ROAD_CENTER_BIAS - e * frame.shape[1] / 2:.0f}"
              f"   (현재 {ROAD_CENTER_BIAS})  ※ 도로 한가운데 똑바로 세운 상태에서만")

    vis = frame.copy()
    vis[road_mask(frame) > 0] = (0, 0, 255)
    cv2.imwrite("road_check.jpg", vis)
    print("road_check.jpg 저장 (빨강 = 도로로 인식한 영역)")
    return 0


ACTION_NAMES = {GO_STRAIGHT: "전진", MOVE_RIGHT: "우회전", MOVE_LEFT: "좌회전",
                GO_BACKWARD: "후진", APPLE_COUNT_ACTION: "사과 세기",
                GO_TO_MARKER: "마커까지 전진",
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


CMDS = {"run": cmd_run, "run2": cmd_run2, "check": cmd_check, "pose": cmd_pose,
        "motors": cmd_motors, "track": cmd_track, "actions": cmd_actions,
        "forward-cal": cmd_forward_cal, "turn-cal": cmd_turn_cal,
        "road": cmd_road, "cal": cmd_cal}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd not in CMDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(CMDS[cmd](*sys.argv[2:]) or 0)
