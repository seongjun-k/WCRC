# =========================================================
# 2026 WCRC 학생용 템플릿 코드 (Pinky pro)
# ---------------------------------------------------------
# 이 코드는 전체 골격(프레임워크)만 제공합니다.
# target_list / after_track_list 에는 "예시 2개"만 들어 있습니다.
# 실제 맵에 맞는 좌표와 동작은 여러분이 직접 측정하고 설계해서
# 채워 넣어야 합니다.
#
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★  학생이 수정하는 곳은 아래 "4곳" 뿐입니다.
# ★
# ★   [학생 수정 ①] my_ip                : Flask 서버 PC의 IP 주소
# ★   [학생 수정 ②] MOVE_FORWARD_PER_ONE : 1초 전진 시 z 변화량 (직접 측정)
# ★   [학생 수정 ③] target_list          : 주행 순서 + 마커 좌표
# ★   [학생 수정 ④] after_track_list     : 마커 도착 후 수행할 동작
# ★
# ★  → Ctrl+F 로 "학생 수정" 을 검색하면 4곳을 바로 찾아갈 수 있습니다.
# ★  → 나머지 코드는 수정하지 않습니다. (읽고 이해하는 것이 목표!)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
#
# 실행 전 준비 (순서 중요):
#   1) PC에서 run.bat 으로 Flask 서버를 먼저 실행하고
#      브라우저에서 .pt 모델 파일을 등록합니다.
#   2) my_ip 에 PC의 IP 주소를 입력합니다. (cmd → ipconfig)
#   3) 로봇의 jupyter notebook 에서 이 코드를 실행합니다.
# =========================================================

import cv2
import time
import requests
from enum import Enum
from pinkylib import Motor
from pinkylib import Buzzer
from pinkylib import Camera
from pinkylib.yolo import Yolo
from collections import Counter
from pinky_lcd.pinky_lcd import LCD
from PIL import Image, ImageSequence, ImageDraw, ImageFont


# ---------------------------------------------------------
# 상수 정의 (수정 ×)
#  - 단, MOTOR_SPEED / SEARCH_MOTOR_SPEED / TURN_..._TIME 은
#    로봇 상태에 따라 "선택적으로" 조정할 수 있습니다.
#    (먼저 기본값으로 테스트해 본 뒤 필요할 때만 바꾸세요)
# ---------------------------------------------------------
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
CROSS_WALK_WAIT = 106     # 잠시 대기 (예: 횡단보도)
DEFAULT = 999             # 시간 옵션이 필요 없는 동작에 사용


MOTOR_SPEED = 90
SEARCH_MOTOR_SPEED = 65

SEARCH_TURN_TIME = 0.05
MATCH_TURN_TIME = 0.05
MATCH_FORWARD_TIME = 0.4

SLEEP_TIME_AFTER_MOVE = 0.15
MOTOR_BIG_STEP_FORWARD = 1
TURN_HALF_TIME = 2              # 모터 스피드에 따라 변경 필요할 수 있음
TURN_QUATER_TIME = 1            # 모터 스피드에 따라 변경 필요할 수 있음
STRAIGHT_TO_MAIN_ROAD_TIME = 3  # 모터 스피드에 따라 변경 필요할 수 있음


# 사과 갯수가 담기는 전역 변수
# (빨간 사과, 초록 사과를 구분하지 않고 읽은 총 갯수)
total_apple_count = 0


# =========================================================
# 사용자 설정 영역  ★ 여기부터 "학생 수정 구역"이 시작됩니다 ★
# =========================================================

SEARCH_COUNT = 5          # 아루코 마커 찾을 횟수 (기본값 사용)
APPLE_CHECK_COUNT = 3     # 사과 이미지 flask 서버에 보낼 횟수 (기본값 사용)


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ★ [학생 수정 ①] my_ip — Flask 서버가 실행 중인 PC의 IP 주소
#    (Windows cmd 창에서 ipconfig 로 확인)
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

my_ip = "192.168.x.x"     # ← ★ 본인 PC의 IP 주소로 변경하세요

# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [학생 수정 ①] 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ★ [학생 수정 ②] MOVE_FORWARD_PER_ONE
#    time.sleep(1) 초 동안 전진했을 때의 아루코 마커 z축 변화 값.
#    로봇마다 다르므로 반드시 본인 로봇으로 직접 측정해서 수정하세요.
#    (측정 코드는 이 파일 맨 아래 "부록" 참고. 음수면 절댓값 사용)
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

MOVE_FORWARD_PER_ONE = 43.792   # ← ★ 본인 로봇으로 측정한 값으로 변경하세요

# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [학생 수정 ②] 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ★ [학생 수정 ③] target_list — 아루코마커 주행 순서 id 리스트
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# pose : [x, z, direction_to_search]
#   x : 마커 앞에 정렬을 마쳤을 때의 목표 x 값 (카메라 기준 좌우 위치)
#   z : 마커 앞에 멈출 때의 목표 z 값 (마커까지의 거리)
#   direction_to_search : 마커 탐색을 시작할 회전 방향 (LEFT / RIGHT)
#
# 아래 2개는 형식을 보여주기 위한 "예시"입니다.
# 실제 맵에서 마커 앞에 로봇을 세워 두고 x, z 값을 측정한 뒤,
# 주행 순서대로 항목을 직접 추가하세요.
target_list = [
    {"id": 1, "pose": [12.56, 55, RIGHT]},   # 예시 1 (★ 실제 맵 값으로 교체)
    {"id": 3, "pose": [6, 75, RIGHT]},       # 예시 2 (★ 실제 맵 값으로 교체)
    # {"id": ?, "pose": [?, ?, ?]},          # ← ★ 여기서부터 직접 추가
]
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [학생 수정 ③] 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ★ [학생 수정 ④] after_track_list — 마커 도착 후 수행할 동작 리스트
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# target_list 와 "같은 순서, 같은 개수"로 작성해야 합니다.
# (i번째 마커에 도착하면 i번째 actions 가 순서대로 실행됩니다)
#
# 사용할 수 있는 동작:
#   (GO_STRAIGHT, 초)             : 전진  (초 = 얼마나 전진할지)
#   (GO_BACKWARD, 초)             : 후진
#   (MOVE_LEFT, 초)               : 좌회전
#   (MOVE_RIGHT, 초)              : 우회전
#   (APPLE_COUNT_ACTION, DEFAULT) : 사과 개수 세기 (서버로 이미지 전송)
#   (APPLE_DISPLAY, DEFAULT)      : LCD에 누적 사과 개수 표시
#   (CROSS_WALK_WAIT, DEFAULT)    : 잠시 대기
# 아무 동작이 없을 경우 빈 칸으로 입력해줍니다:
#   {"id": 0, "actions": []}
#
# 아래 2개 예시를 참고해 미션에 맞게 직접 설계하세요.
after_track_list = [
    {"id": 1, "actions": [(MOVE_RIGHT, 0.4), (GO_STRAIGHT, 0.3)]},                    # 예시 1: 주행 동작만 (★ 직접 설계)
    {"id": 3, "actions": [(APPLE_COUNT_ACTION, DEFAULT), (APPLE_DISPLAY, DEFAULT)]},  # 예시 2: 사과 세기 + LCD 표시 (★ 직접 설계)
    # {"id": ?, "actions": []},   # ← ★ 여기서부터 직접 추가
]
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [학생 수정 ④] 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# =========================================================
# ★ 학생 수정 구역 끝! ★
# 여기서부터 파일 끝까지는 "수정하지 않는" 영역입니다.
# (코드를 읽으면서 전체 동작 흐름을 이해하는 것이 목표입니다)
# =========================================================

SEARCH_RETURN_TIME = SEARCH_TURN_TIME * SEARCH_COUNT

# ---------------------------------------------------------
# 하드웨어 초기화
# ---------------------------------------------------------
pinky_buzzer = Buzzer()
pinky_motor = Motor()
pinky_cam = Camera()
pinky_lcd = LCD()

pinky_cam.set_calibration()

pinky_cam.start()
pinky_motor.enable_motor()
pinky_buzzer.buzzer_start()


# ---------------------------------------------------------
# Flask 서버 통신 (사과 개수 받아오기)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# LCD 표시 & 사과 카운트
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 아루코 마커 인식 / 탐색
# ---------------------------------------------------------
def detect_target_aruco(aruco_num):
    frame = pinky_cam.get_frame()
    output_frame, pose = pinky_cam.detect_aruco(frame, marker_size=0.1)

    if pose is None:
        print("None is detected")
        return False, None
    elif pose[0][0] != aruco_num:
        print("Target not detected")
        print("id: ", str(pose[0][0]), "x: ", str(pose[0][1]), "y: ", str(pose[0][2]), "z: ", str(pose[0][3]))
        return False, None
    else:
        print("Target detected")
        print("id: ", str(pose[0][0]), "x: ", str(pose[0][1]), "y: ", str(pose[0][2]), "z: ", str(pose[0][3]))
        return True, pose


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
                    move_left(SEARCH_TURN_TIME, SEARCH_MOTOR_SPEED)
                    print("left, not detected")
                else:
                    move_right(SEARCH_TURN_TIME, SEARCH_MOTOR_SPEED)
                    print("right, not detected")
                print("i", i)
                time.sleep(SLEEP_TIME_AFTER_MOVE)
        # try_count 번 다 돌아도 못 찾으면 원래 방향으로 복귀
        if direction == LEFT:
            move_right(SEARCH_RETURN_TIME)
        else:
            move_left(SEARCH_RETURN_TIME)
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
                move_left(SEARCH_TURN_TIME, SEARCH_MOTOR_SPEED)
                print("left, no limit, not detected")
            else:
                move_right(SEARCH_TURN_TIME, SEARCH_MOTOR_SPEED)
                print("right, no limit, not detected")
            time.sleep(SLEEP_TIME_AFTER_MOVE)


def find_aruco(aruco_num, direction, try_count):
    # 지정한 방향으로 try_count 번 탐색하고, 실패하면 반대 방향으로 한 번 더 탐색합니다.
    result, pose = find_aruco_with_try_count(aruco_num, direction, try_count)
    final_result = 0
    final_pose = None

    if (result != 1):
        if direction == LEFT:
            time.sleep(SLEEP_TIME_AFTER_MOVE)
            final_result, final_pose = find_aruco_with_try_count(aruco_num, RIGHT, try_count)
        elif direction == RIGHT:
            time.sleep(SLEEP_TIME_AFTER_MOVE)
            final_result, final_pose = find_aruco_with_try_count(aruco_num, LEFT, try_count)
    else:
        return result, pose

    return final_result, final_pose


# ---------------------------------------------------------
# 각도 / 거리 정렬 & 마커 추적
# ---------------------------------------------------------
def check_angle(aruco_num, target, allow_range=15):
    success, pose = detect_target_aruco(aruco_num)
    if success:
        cur_pose_x = pose[0][1]
        print("current_pose_x", cur_pose_x, "target_pose_x", target)
        if (target - allow_range < cur_pose_x) and (cur_pose_x < target + allow_range):
            return True, 0
        elif cur_pose_x < target - allow_range:
            return False, LEFT
        elif target + allow_range < cur_pose_x:
            return False, RIGHT
    else:
        print("check_angle, not detected")
        return False, None


def check_distance(aruco_num, target, allow_range=5):
    success, pose = detect_target_aruco(aruco_num)
    if success:
        cur_pose_z = pose[0][3]
        print("cur_pose_z", cur_pose_z, "target_pose_z", target)
        if (target - allow_range <= cur_pose_z) and (cur_pose_z <= target + allow_range):
            return True, None
        elif target + allow_range < cur_pose_z:
            # 아직 멀다 -> 남은 거리를 시간으로 환산해서 전진
            temp_time = abs(target - cur_pose_z) / MOVE_FORWARD_PER_ONE
            print("temp_time", temp_time)
            return False, temp_time
        elif target - allow_range > cur_pose_z:
            print("check_distance, over the target")
            return True, None
    else:
        print("check_distance, not detected")
        return False, None


def track_target_aruco_marker(aruco_num, target_pose, try_count=0):
    # 1) 마커 탐색 -> 2) 각도(x) 정렬 -> 3) 거리(z) 접근
    target_x, target_z, target_direction = target_pose

    aruco_success, pose = find_aruco(aruco_num, target_direction, try_count)
    if aruco_success:

        final_result = False

        # 각도(x) 맞추기
        while not final_result:
            final_result, angle_direction = check_angle(aruco_num, target_x)
            print("check_angle : direction  ", angle_direction)
            if not final_result:
                print("angle not ready")
                if angle_direction == LEFT:
                    move_left(MATCH_TURN_TIME)
                    time.sleep(SLEEP_TIME_AFTER_MOVE)
                else:
                    move_right(MATCH_TURN_TIME)
                    time.sleep(SLEEP_TIME_AFTER_MOVE)
        print("angle success")

        final_distance_result = False
        not_detected_count = 0

        # 거리(z) 맞추기
        while not final_distance_result:
            final_distance_result, distance_direction = check_distance(aruco_num, target_z)
            if not final_distance_result:
                print("distance not ready")
                if distance_direction is not None:
                    move_forward(distance_direction)
                    time.sleep(SLEEP_TIME_AFTER_MOVE)
                else:
                    not_detected_count += 1
                    print("distance detection failed")
                    if not_detected_count >= 3:
                        not_detected_count = 0
                        break
        return True

    else:
        print("track_target_aruco_marker find_aruco_failed")
        return False


# ---------------------------------------------------------
# 모터 동작 함수
# ---------------------------------------------------------
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


def go_straight_to_main_road(duration_time=STRAIGHT_TO_MAIN_ROAD_TIME):
    motor_speed = MOTOR_SPEED
    move_forward(duration_time)
    time.sleep(duration_time)
    pinky_motor.move(0, 0)


# ---------------------------------------------------------
# 마커 도착 후 동작 실행기
# ---------------------------------------------------------
def after_target_do_list(index):
    global total_apple_count
    action = after_track_list[index]
    current_action = action["actions"]
    if len(current_action) == 0:
        print("nothing")
        return
    for i in range(len(current_action)):
        action_inside, option_inside = current_action[i]
        if action_inside == GO_STRAIGHT:
            move_forward(option_inside)
        elif action_inside == MOVE_RIGHT:
            move_right(option_inside)
        elif action_inside == MOVE_LEFT:
            move_left(option_inside)
        elif action_inside == GO_BACKWARD:
            move_backward(option_inside)
        elif action_inside == APPLE_DISPLAY:
            display_apple_count(total_apple_count)
        elif action_inside == APPLE_COUNT_ACTION:
            temp_apple_count = predict_apple_count()
            total_apple_count = total_apple_count + temp_apple_count
        elif action_inside == CROSS_WALK_WAIT:
            time.sleep(0.5)
        else:
            print("wrong_action")


# =========================================================
# 메인 실행 루프
# ---------------------------------------------------------
# target_list 의 마커를 순서대로 추적하고,
# 각 마커에 도착할 때마다 after_track_list 의 동작을 실행합니다.
# 주의: 이 부분을 실행하면 로봇이 실제로 주행합니다!
# =========================================================

SERVER_URL = get_server_url(my_ip)

assert len(target_list) == len(after_track_list), \
    "target_list 와 after_track_list 의 개수가 다릅니다! 같은 순서/개수로 맞춰주세요."

for i in range(len(target_list)):
    current_id = target_list[i]["id"]
    current_target_pose = target_list[i]["pose"]
    target_x, target_z, target_direction = current_target_pose
    result = track_target_aruco_marker(current_id, current_target_pose, SEARCH_COUNT)
    if result is not True:
        break
    after_target_do_list(i)
    print("----list num : ", i, " done -----")


time.sleep(10)


# ---------------------------------------------------------
# 종료 (하드웨어 해제)
# ---------------------------------------------------------
pinky_cam.close()
pinky_buzzer.close()
pinky_motor.close()
pinky_lcd.close()


# =========================================================
# (부록) MOVE_FORWARD_PER_ONE 측정 코드  (실행용 — 코드 수정 ×)
#  → [학생 수정 ②] 에 넣을 값을 구할 때 사용합니다.
# ---------------------------------------------------------
# 사용법:
#   1) 위의 "메인 실행 루프" for문 전체를 주석 처리한다
#   2) 로봇을 아루코 마커가 정면에 보이는 위치에 놓는다
#   3) 아래 주석(''')을 해제하고 실행한다
#   4) 출력되는 rate 값을 MOVE_FORWARD_PER_ONE 에 입력한다
#      (rate 가 음수로 나오면 절댓값을 사용)
# 주의: detect_aruco 의 marker_size 는 위 detect_target_aruco 와
#       동일한 0.1 을 사용해야 z 값의 스케일이 맞습니다.
# =========================================================
'''
frame = pinky_cam.get_frame()
output_frame, pose = pinky_cam.detect_aruco(frame, marker_size=0.1)
pinky_cam.display_jupyter(output_frame)
print("id: ", str(pose[0][0]), "x: ", str(pose[0][1]), "y: ", str(pose[0][2]), "z: ", str(pose[0][3]))

z_1 = pose[0][3]

move_forward(1)
time.sleep(0.5)

frame = pinky_cam.get_frame()
output_frame, pose = pinky_cam.detect_aruco(frame, marker_size=0.1)
pinky_cam.display_jupyter(output_frame)
print("id: ", str(pose[0][0]), "x: ", str(pose[0][1]), "y: ", str(pose[0][2]), "z: ", str(pose[0][3]))

z_2 = pose[0][3]

print("z_1: ", str(z_1), "z_2: ", str(z_2))

rate = z_2 - z_1

print("rate: ", rate)
'''
