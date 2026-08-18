"""과수원 앞에서 학습용 사진을 모은다. 로봇에서 실행한다.

    python3 capture_sweep.py [태그]        # 태그로 과수원 A/B/C 구분

이동 제약: 전진은 최대 5cm 까지만 하고 반드시 되돌아온다.
변화는 대부분 제자리 회전으로 만든다 (회전은 병진이 없어 자리를 안 벗어난다).

★ 찍기 전에 나무 뒤를 비울 것.
  직전 학습이 실패한 이유가 이거였다 — 99장 전부 나무 뒤에 사람이 앉아 있어서
  모델이 "빨간 사과"가 아니라 "밝은 배경 위의 빨간 덩어리"를 배웠다.
  대회 때는 그 자리에 사람이 없으므로 학습 배경도 그래야 한다.
"""
import os, sys, time
import cv2
sys.path.insert(0, '/home/pinky')
import drive

TAG   = sys.argv[1] if len(sys.argv) > 1 else "a"
OUT   = "/home/pinky/raw/images"
NEAR  = 5.0                                  # 전진 허용 한계(cm)
ANGLES = list(range(-20, 21, 4))             # -20 ~ +20 도, 4도 간격 (11자세)
SHOTS  = 4
NEG    = 20

os.makedirs(OUT, exist_ok=True)
drive.setup(motors=True)
n = 0
def shot():
    global n
    cv2.imwrite(f"{OUT}/orch{TAG}_{n:04d}.jpg", drive.pinky_cam.get_frame())
    n += 1
    time.sleep(0.12)
try:
    for d in (0.0, NEAR):
        if d:
            drive.move_forward(d / drive.MOVE_FORWARD_PER_ONE)
            time.sleep(0.5)
        turned = 0.0
        for a in ANGLES:
            drive.turn_deg(a - turned); turned = a
            time.sleep(0.35)
            for _ in range(SHOTS):
                shot()
            print(f"  전진 {d:.0f}cm  각도 {a:+3}도  누적 {n}장")
        drive.turn_deg(-turned)              # 각도 원위치
        time.sleep(0.4)
        if d:
            drive.move_backward(d / drive.MOVE_FORWARD_PER_ONE)   # 거리도 원위치
            time.sleep(0.5)

    # 배경(네거티브): 사과가 안 보이는 방향. 오탐을 눌러준다.
    print("배경 사진 (사과 없음)")
    swept = 0
    drive.turn_deg(70); swept += 70
    for k in range(NEG):
        shot()
        if k % 5 == 4:
            drive.turn_deg(20); swept += 20
            time.sleep(0.35)
    drive.turn_deg(-swept)
    print(f"\n총 {n}장 -> {OUT}/orch{TAG}_*.jpg")
finally:
    drive.teardown()
