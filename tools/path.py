"""`drive.py run2` 의 외운 경로 검증.

구현은 drive.py 에만 있다. 여기서는 가짜 로봇(바퀴가 안 미끄러지고 IMU 가 정확한
이상적인 로봇)에 PATH_LEGS 를 태워, 실제로 그 형상대로 움직이는지 본다.

    python tools/path.py            # 자체 점검
    python tools/path.py --plot     # 구간별 궤적을 그림으로 (scratchpad 에 저장)

경로 원본은 26wcrc_final.pdf 3페이지 도면이다. 뽑는 과정은 메모리
wcrc-drive-by-memorized-path 에 적어두었다.
"""
import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import drive


def fake(m, imu=True, trim_err=0.0):
    """이상적인 로봇. trim_err 를 주면 그만큼 좌우가 치우친 로봇이 된다."""
    st = {"t": 0.0, "x": 0.0, "y": 0.0, "th": 0.0, "v": 0.0, "w": 0.0, "log": []}

    def advance(dt):
        st["x"] += st["v"] * math.cos(math.radians(st["th"])) * dt
        st["y"] += st["v"] * math.sin(math.radians(st["th"])) * dt
        st["th"] += st["w"] * dt
        st["t"] += dt
        if st["v"] or st["w"]:
            st["log"].append((st["x"], st["y"], st["th"]))

    def move(l, r):
        if l == 0 and r == 0:
            st["v"] = st["w"] = 0.0
            return
        st["v"] = m.MOVE_FORWARD_PER_ONE * (l + r) / 2 / m.MOTOR_SPEED
        d = (l - r) / 2 - trim_err
        st["w"] = m.TURN_DEG_PER_SEC * d / m.MOTOR_SPEED

    m.time = types.SimpleNamespace(sleep=advance, time=lambda: st["t"])
    m.pinky_motor = types.SimpleNamespace(move=move)
    m.read_yaw = (lambda: (st["th"] + 180) % 360 - 180) if imu else (lambda: None)
    return st


def run_leg(i, cm, gains=None, **kw):
    """구간 i 를 cm 만큼 달리고 (궤적, 최종 헤딩) 을 돌려준다.

    gains 로 추종기 상수를 덮어쓸 수 있다 (reload 뒤에 물려야 한다 — 밖에서
    drive 에 직접 넣으면 여기 reload 가 도로 지운다).
    """
    import importlib
    m = importlib.reload(drive)
    for k, v in (gains or {}).items():
        setattr(m, k, v)
    st = fake(m, **kw)
    m.leg_begin(i)
    m.follow_path(cm / m.MOVE_FORWARD_PER_ONE)
    return m, st


def want_xy(leg, cm, n=400):
    """프로파일이 뜻하는 이상적인 궤적 (헤딩을 그대로 적분한 것)."""
    ss = np.linspace(0, cm, n)
    hd = np.interp(ss, [p[0] for p in leg], [p[1] for p in leg])
    d = np.diff(ss)
    x = np.concatenate([[0], np.cumsum(np.cos(np.radians(hd[:-1])) * d)])
    y = np.concatenate([[0], np.cumsum(np.sin(np.radians(hd[:-1])) * d)])
    return x, y


def deviation(st, leg, cm):
    """가짜 로봇이 실제로 그린 궤적과 프로파일이 뜻한 궤적의 최대 거리(cm)."""
    gx, gy = want_xy(leg, cm)
    got = np.array([(x, y) for x, y, _ in st["log"]])
    if len(got) < 2:
        return 0.0
    return max(np.hypot(gx - x, gy - y).min() for x, y in got)


def main():
    legs = drive.PATH_LEGS
    print(f"구간 {len(legs)}개")
    worst = 0.0
    for i, leg in enumerate(legs):
        cm = leg[-1][0]
        m, st = run_leg(i, cm)
        dev = deviation(st, leg, cm)
        worst = max(worst, dev)
        end = st["log"][-1][2]
        assert abs(end - leg[-1][1]) < 3.0, \
            f"구간 {i}: 최종 헤딩 {end:+.1f}도 (프로파일은 {leg[-1][1]:+.0f}도)"
        # 좌우 여유는 2.1cm (도로 154mm - 바퀴 111mm). 그 안에 들면 된다.
        assert dev < 1.8, f"구간 {i}: 경로 이탈 {dev:.2f}cm"
        print(f"  [{i}] {cm:>4.0f}cm  헤딩 {end:+7.1f}도 (목표 {leg[-1][1]:+.0f})  "
              f"이탈 {dev:.2f}cm  {m.time.time():5.1f}초")
    print(f"최대 이탈 {worst:.2f}cm (도로 좌우 여유 2.1cm)")

    # 치우친 로봇도 IMU 로 잡아야 한다. 이게 안 되면 카메라를 뗀 의미가 없다.
    i = 4
    leg = legs[i]; cm = leg[-1][0]
    _, bad = run_leg(i, cm, trim_err=4.0)
    dev_fb = deviation(bad, leg, cm)
    _, blind = run_leg(i, cm, imu=False, trim_err=4.0)
    dev_ff = deviation(blind, leg, cm)
    print(f"좌우 편향 4 인 로봇: IMU 닫음 {dev_fb:.2f}cm / 열린루프 {dev_ff:.2f}cm")
    assert dev_fb < 2.1, f"IMU 를 닫아도 도로를 나간다 ({dev_fb:.2f}cm)"
    assert dev_fb < dev_ff, "IMU 피드백이 열린 루프보다 나아지지 않았다"

    # 마커는 로봇을 못 돌린다. 마커 x 하나로 방향과 좌우 위치를 동시에 못 맞추는데
    # 예전 코드는 제자리 회전으로 x 를 맞춰서, 그 회전이 외운 경로를 통째로
    # 망가뜨렸다 (실주행에서 28.8도 돌았다).
    import importlib
    m = importlib.reload(drive)
    assert m.PATH_NO_MARKER_TURN, "마커 회전 차단이 꺼져 있다"
    assert m.PATH_MARKER_CONFIRM_ONLY, "마커가 확인 전용이 아니다"
    m.pinky_cam = types.SimpleNamespace(
        get_frame=lambda: None,
        detect_aruco=lambda f, marker_size=None: (None, [[1, 30.0, 0.0, 45.0]]))
    aligned, deg = m.check_angle(1, -17)     # 오차 47cm — 원래라면 크게 돌 상황
    assert aligned and deg == 0.0, f"마커가 회전을 시킨다 ({deg})"
    print("마커 회전 차단 OK")

    if "--plot" in sys.argv:
        import cv2
        sp = os.environ.get("SP", ".")
        for i, leg in enumerate(legs):
            cm = leg[-1][0]
            _, st = run_leg(i, cm)
            gx, gy = want_xy(leg, cm)
            img = np.zeros((600, 900, 3), np.uint8)
            def to(x, y):
                return int(450 + x * 7), int(300 + y * 7)
            for a, b in zip(zip(gx, gy), list(zip(gx, gy))[1:]):
                cv2.line(img, to(*a), to(*b), (0, 200, 0), 3)
            pts = [(x, y) for x, y, _ in st["log"]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, to(*a), to(*b), (0, 0, 255), 1)
            cv2.imwrite(f"{sp}/leg{i}.png", img)
        print("그림 저장")
    print("\nOK")


if __name__ == "__main__":
    main()
