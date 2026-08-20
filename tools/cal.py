"""`drive.py cal` 검증. 가짜 로봇에 정답 상수를 심어놓고, cal 이 그 값을 되찾아
파일에 제대로 적는지 본다.

    python tools/cal.py

대회장에서 처음 돌려보는 게 진짜 로봇이면 늦다. 여기서 먼저 깨져야 한다.
구현은 drive.py 에만 있다 — 이 파일은 흉내내지 않는다.
"""
import io
import os
import shutil
import sys
import tempfile
import types

import numpy as np

# 같은 초에 두 번 쓴 파일을 importlib 이 stale .pyc 로 읽는다 -> 수렴 테스트가 거짓 통과한다
sys.dont_write_bytecode = True
sys.stdin = io.StringIO("\n" * 200)   # 대화형 확인은 엔터를 친 것으로 본다

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import drive

TRUE_CMPS = 21.5      # 센서를 떼서 가벼워진 로봇이라 치자 (현재 상수 17.72 와 다른 값)
TRUE_DPS = 245.0      # MOTOR_SPEED 에서의 각속도
TRUE_COAST = 0.190    # 도 / 속도
TRUE_TRIM = -3        # 이만큼 보정해야 똑바로 가는 로봇 (왼쪽이 느리다)


def fake_robot(m, marker_z=55.0):
    """m(=drive 사본)의 하드웨어를 가짜로 갈아끼운다. 시계도 가짜다."""
    st = {"t": 0.0, "yaw": 0.0, "pos": 0.0, "w": 0.0, "v": 0.0, "coast": 0.0}

    def advance(dt):
        while dt > 1e-12:
            if st["coast"] > 0:                    # 정지 명령 뒤 관성으로 더 도는 구간
                step = min(dt, st["coast"])
                st["yaw"] += st["w"] * step
                st["coast"] -= step
                dt -= step
                if st["coast"] <= 1e-12:
                    st["w"] = 0.0
            else:
                st["yaw"] += st["w"] * dt
                st["pos"] += st["v"] * dt
                dt = 0.0
        st["t"] += 0.0

    def sleep(s):
        st["t"] += s
        advance(s)

    def move(l, r):
        if l == 0 and r == 0:
            if st["w"]:                            # 회전 중이었다면 coast 를 남긴다
                speed = abs(st["w"]) * m.MOTOR_SPEED / TRUE_DPS
                st["coast"] = TRUE_COAST * speed / abs(st["w"])
            st["v"] = 0.0
        elif l * r > 0 and abs(l - r) < abs(l + r) / 2:   # 직진 (trim 만큼 좌우가 다르다)
            st["v"] = TRUE_CMPS * (l + r) / 2 / m.MOTOR_SPEED
            # 좌우 차가 TRUE_TRIM 만큼일 때 안 휜다. 회전과 같은 환산식을 쓴다.
            d = (l - r) / 2 - (TRUE_TRIM if l > 0 else -TRUE_TRIM)
            st["w"], st["coast"] = TRUE_DPS * d / m.MOTOR_SPEED, 0.0
        else:
            speed = abs(l)
            st["v"], st["coast"] = 0.0, 0.0
            st["w"] = TRUE_DPS * speed / m.MOTOR_SPEED * (1 if l > 0 else -1)

    def read_yaw():
        advance(0.004)                             # IMU 한 번 읽는 데 걸리는 시간
        st["t"] += 0.004
        return (st["yaw"] + 180) % 360 - 180

    m.time = types.SimpleNamespace(sleep=sleep, time=lambda: st["t"])
    m.pinky_motor = types.SimpleNamespace(move=move)
    m.pinky_cam = types.SimpleNamespace(
        get_frame=lambda: np.zeros((240, 320, 3), np.uint8),
        detect_aruco=lambda f, marker_size=None: (True, [[1, 0.0, 0.0, marker_z - st["pos"]]]))
    m.read_yaw = read_yaw
    m.setup = lambda motors=True: None
    m.teardown = lambda: None
    m.road_offset = lambda f: (-0.10, 0.0)         # 카메라가 왼쪽으로 치우친 상태
    return st


_n = [0]


def load(path):
    """path 파일을 그대로 읽어 가짜 하드웨어를 물린다.

    reload(drive) 가 아니다 — 그러면 원본을 읽어서, cal 이 방금 기록한 값이
    다음 회차에 반영되지 않는다. 수렴을 확인하려면 쓴 파일을 다시 읽어야 한다.
    """
    import importlib.util
    _n[0] += 1
    spec = importlib.util.spec_from_file_location(f"drive_cal{_n[0]}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.__file__ = path
    fake_robot(m)
    return m


def main():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "drive.py")
    shutil.copy(drive.__file__, path)
    m = load(path)

    assert m.cmd_cal("--yes") == 0, "cal 이 0 이 아닌 값을 냈다"

    got = {}
    for line in open(path, encoding="utf-8"):
        for k in ("MOVE_FORWARD_PER_ONE", "TURN_DEG_PER_SEC", "MOTOR_TRIM",
                  "TURN_COAST_PER_SPEED", "ROAD_CENTER_BIAS"):
            if line.startswith(k + " = "):
                got.setdefault(k, float(line.split("=")[1].split("#")[0]))

    assert len(got) == 5, f"파일에 안 적힌 상수가 있다: {got}"
    assert abs(got["MOVE_FORWARD_PER_ONE"] - TRUE_CMPS) < 0.3, got
    assert abs(got["TURN_DEG_PER_SEC"] - TRUE_DPS) < 5, got
    assert abs(got["TURN_COAST_PER_SPEED"] - TRUE_COAST) < 0.01, got
    # error -0.10, 폭 320 -> 중심이 16px 오른쪽에 있다고 봐야 한다
    assert got["ROAD_CENTER_BIAS"] == 16, got
    print("cal: 네 상수 모두 정답 복원 + 파일 기록 OK", got)

    # 백업이 있어야 되돌릴 수 있다
    assert os.path.exists(path + ".bak"), "백업을 안 만들었다"
    assert open(path + ".bak", encoding="utf-8").read() == \
        open(drive.__file__, encoding="utf-8").read(), "백업이 원본과 다르다"
    print("백업 OK")

    # 없는 상수는 조용히 넘어가면 안 된다 — 오타 하나로 캘리브레이션이 무효가 된다
    try:
        m._write_const("NOPE_NOT_A_CONST", 1, path)
    except KeyError:
        print("없는 상수 -> 예외 OK")
    else:
        raise AssertionError("없는 상수인데 그냥 넘어갔다")

    # MOTOR_TRIM 은 한 번에 안 잡힌다 (정지 관성이 섞인다). 반복하면 수렴해야 한다.
    trims = [got["MOTOR_TRIM"]]
    for _ in range(4):
        mm = load(path)
        assert mm.cmd_cal("forward", "--yes") == 0
        trims.append(next(float(l.split("=")[1].split("#")[0])
                          for l in open(path, encoding="utf-8") if l.startswith("MOTOR_TRIM = ")))
    assert abs(trims[-1] - TRUE_TRIM) <= 1, f"MOTOR_TRIM 이 수렴 안 했다: {trims}"
    assert abs(trims[-1] - TRUE_TRIM) < abs(trims[0] - TRUE_TRIM) + 0.01, \
        f"반복해도 나아지지 않는다: {trims}"
    print(f"MOTOR_TRIM 수렴 OK {trims} -> 정답 {TRUE_TRIM}")

    # 골라서 재기
    m2 = load(path)
    assert m2.cmd_cal("road", "--yes") == 0
    print("부분 실행 OK")

    shutil.rmtree(tmp)
    print("\nOK")


if __name__ == "__main__":
    main()
