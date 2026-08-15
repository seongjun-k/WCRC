"""도로 유지 주행 튜닝·검증 도구.

구현은 여기 없다. **drive.py 가 원본**이고, 이 파일은 거기서 함수를 그대로 가져와
실제 사진으로 시험한다. 코드가 두 벌로 갈라지지 않게 하려는 것이다.

    python tools/road.py                    # 자체 점검 (raw/images)
    python tools/road.py <이미지>           # 한 장 진단 + 디버그 이미지 저장
    python tools/road.py --tune <이미지>    # 그 사진에 맞는 S/V 임계값 추천

⚠️ ROAD_S_MAX / ROAD_V_MIN 은 조명에 민감하다. 대회장에서 로봇을 **도로 위에
도로를 따라 보게** 놓고 찍은 사진으로 --tune 을 돌려 drive.py 값을 고칠 것.
로봇에서 바로 볼 수도 있다:  python3 drive.py road
"""
import glob
import os
import sys
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import drive


def load():
    """drive 를 매번 새로 읽어 가짜 하드웨어를 물린 사본을 돌려준다."""
    import importlib
    m = importlib.reload(drive)
    m.time = types.SimpleNamespace(sleep=lambda s: None)
    return m


road_mask = drive.road_mask
road_offset = drive.road_offset
road_steer = drive.road_steer
R = {k: getattr(drive, k) for k in
     ("ROAD_S_MAX", "ROAD_V_MIN", "ROAD_NEAR_BAND", "ROAD_FAR_BAND", "road_band_center")}


def tune(path):
    frame = cv2.imread(path)
    assert frame is not None, path
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = frame.shape[0]
    low = hsv[int(h * 0.8):].reshape(-1, 3)     # 바로 앞 = 대부분 도로일 것
    s, v = low[:, 1], low[:, 2]
    print(f"{os.path.basename(path)} 아래 20% 통계")
    print(f"  S 중앙값 {np.median(s):.0f}   75%tile {np.percentile(s, 75):.0f}")
    print(f"  V 중앙값 {np.median(v):.0f}   25%tile {np.percentile(v, 25):.0f}")
    s_max = np.percentile(s, 60) + 15
    v_min = max(0, np.percentile(v, 30) - 15)
    print(f"\ndrive.py 에 넣을 값:")
    print(f"  ROAD_S_MAX = {s_max:.0f}")
    print(f"  ROAD_V_MIN = {v_min:.0f}")
    if s_max > 110:
        print("\n  ⚠️ S_MAX 추천치가 너무 높다. 이 사진은 아래쪽이 도로가 아닌 것 같다")
        print("     (잔디 채도가 130 근처라 이대로 쓰면 잔디까지 도로로 본다).")
        print("     로봇을 도로 위에 도로를 따라 보게 놓고 다시 찍을 것.")
    print(f"\n(현재 drive.py 값: ROAD_S_MAX={R['ROAD_S_MAX']} ROAD_V_MIN={R['ROAD_V_MIN']})")


def diagnose(path):
    frame = cv2.imread(path)
    assert frame is not None, path
    e, c = road_offset(frame)
    print(f"{os.path.basename(path)}  error={e}  curve={c}  steer={road_steer(frame)}")
    if e is None:
        print("  도로를 못 찾았다 -> --tune 으로 임계값 확인")

    mask = road_mask(frame)
    vis = frame.copy()
    vis[mask > 0] = (0, 0, 255)
    h, w = frame.shape[:2]
    for band, color in ((R["ROAD_NEAR_BAND"], (255, 0, 255)),
                        (R["ROAD_FAR_BAND"], (0, 255, 255))):
        cx = R["road_band_center"](mask, band)
        cv2.line(vis, (0, int(h * band[0])), (w, int(h * band[0])), color, 1)
        if cx is not None:
            cv2.circle(vis, (int(cx), int(h * (band[0] + band[1]) / 2)), 9, color, -1)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    out = os.path.splitext(path)[0] + "_road.jpg"
    cv2.imwrite(out, vis)
    print("디버그 이미지:", out, "(빨강=도로 마스크, 원=밴드별 도로 중심)")


def _check_move_forward_on_road():
    """전진 분할·바퀴 클램프 검사. 카메라·모터를 가짜로 물려 노트북 함수를 그대로 돌린다."""
    frame = cv2.imread(sorted(glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "raw", "images", "*.jpg")))[0])

    moves, slept, plain = [], [], []
    g = load()
    g.pinky_motor = types.SimpleNamespace(move=lambda l, r: moves.append((l, r)))
    g.pinky_cam = types.SimpleNamespace(get_frame=lambda: frame)
    g.time = types.SimpleNamespace(sleep=slept.append)
    g.move_forward = lambda t, sp=90: plain.append(t)

    # 긴 전진: 0.25초씩 쪼개져야 한다
    g.move_forward_on_road(1.0)
    assert not plain, "긴 전진인데 조향 없는 move_forward 로 빠졌다"
    assert abs(sum(slept) - 1.0) < 1e-9, f"전진 시간 합계가 {sum(slept)} (1.0 이어야)"
    assert len(slept) == 4, f"1.0초가 {len(slept)}조각 (0.25초씩 4조각이어야)"
    driving = [m for m in moves if m != (0, 0)]
    assert driving, "모터에 전진 명령이 없다"
    for l, r in driving:
        assert l > 0 and r > 0, f"전진 중 바퀴가 역회전한다 {(l, r)} — 제자리 회전이 된다"
        assert l <= 100 and r <= 100, f"바퀴 속도 범위 초과 {(l, r)}"
    assert moves[-1] == (0, 0), "마지막에 모터를 안 세웠다"

    # 짧은 전진: 조향하지 않고 그대로 직진
    moves.clear(); slept.clear(); plain.clear()
    g.move_forward_on_road(0.2)
    assert plain == [0.2], f"짧은 전진이 직진으로 안 빠졌다: {plain}"
    assert not moves, "짧은 전진인데 조향 명령이 나갔다"
    print("move_forward_on_road: 분할·클램프·짧은전진 예외 OK")


def demo():
    here = os.path.dirname(__file__)
    paths = sorted(glob.glob(os.path.join(here, "..", "raw", "images", "*.jpg")))
    assert paths, "raw/images 에 사진이 없다"

    found = [(p, *road_offset(cv2.imread(p))) for p in paths]
    found = [(p, e, c) for p, e, c in found if e is not None]
    rate = len(found) / len(paths)
    print(f"도로 인식: {len(found)}/{len(paths)} ({rate:.0%})")
    assert rate > 0.8, f"인식률 {rate:.0%} — --tune 으로 임계값을 다시 잡을 것"

    # 부호 규약: 이미지를 오른쪽으로 밀면 도로도 오른쪽으로 가므로 error 는 줄어야 한다.
    # 사진 내용에 기대지 않는 검사라 조명이 바뀌어도 이 성질은 유지돼야 한다.
    checked = 0
    for p, e, _ in found[:20]:
        f = cv2.imread(p)
        shifted = cv2.warpAffine(f, np.float32([[1, 0, 60], [0, 1, 0]]),
                                 (f.shape[1], f.shape[0]))
        e2, _ = road_offset(shifted)
        if e2 is None:
            continue
        assert e2 <= e + 1e-6, f"{os.path.basename(p)}: 오른쪽으로 밀었는데 error 증가 {e:+.3f}->{e2:+.3f}"
        checked += 1
    assert checked >= 5, f"부호 검사를 {checked}장밖에 못 했다"
    print(f"부호 규약 확인 {checked}장 (오른쪽으로 밀면 error 감소)")

    for p, _, _ in found:
        s = road_steer(cv2.imread(p))
        assert s is None or -1 <= s <= 1, f"{p}: steer={s}"
    print("steer 범위 OK")

    _check_move_forward_on_road()

    print("\nOK — 단, raw/images 는 과수원을 보고 찍은 사진이라 주행 시점이 아니다.")
    print("     대회장에서 도로를 따라 보는 사진으로 --tune 을 반드시 다시 돌릴 것.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--tune":
        tune(sys.argv[2])
    elif len(sys.argv) > 1:
        diagnose(sys.argv[1])
    else:
        demo()
