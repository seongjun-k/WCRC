"""drive.py 의 주행 로직을 하드웨어 없이 돌려보는 시뮬레이터.

pinkylib 대신 가짜 카메라·모터를 물리고 drive.run_course() 를 그대로 실행한다.
가상 로봇은 마커의 방위각(deg)과 거리(cm)만 갖는 1차원 모델이다.

    python tools/sim_drive.py

목적은 "코스를 정확히 재현"이 아니라 **멈추지 않는지** 확인하는 것이다.
규정 패널티 1번(60초 정체 시 기회 종료)에 걸리는 무한루프가 제일 무서운 실패다.
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import drive


class Clock:
    """time.sleep 이 실제로 자지 않고 가상 시간만 민다. 안 그러면 시뮬이 몇 분 걸린다."""
    def __init__(self):
        self.t = 0.0
    def time(self):
        return self.t
    def sleep(self, s):
        self.t += float(s)


CLOCK = Clock()


class World:
    """마커를 방위각(deg)·거리(cm)로 들고 있는다. 로봇은 heading 만 바꾼다."""

    FOV = 30.0            # 카메라 반각
    MAX_RANGE = 150.0     # 이보다 멀면 인식 실패
    DEG_PER_SEC = 212.0   # 실측: MOTOR_SPEED 90 에서 212도/초 (시간에 선형)
    TURN_SLIP = 0.9       # IMU 로 닫으므로 명령 각도의 90% 는 실제로 돈다고 본다
    # 실제 전진 속도. drive.py 상수를 그대로 쓴다 — 예전엔 43.792 로 박아둬서
    # drive.py 를 고쳐도 시뮬이 옛 속도로 돌았다.
    CM_PER_SEC = drive.MOVE_FORWARD_PER_ONE

    def __init__(self):
        self.markers = {}

    def turn(self, deg):
        for m in self.markers.values():
            m[0] -= deg       # 로봇이 돌면 마커의 상대 방위각은 반대로 움직인다

    def forward(self, cm):
        for m in self.markers.values():
            m[1] -= cm * math.cos(math.radians(max(-89, min(89, m[0]))))

    def visible(self):
        out = [[mid, math.tan(math.radians(az)) * dist, 0.0, dist]
               for mid, (az, dist) in self.markers.items()
               if abs(az) <= self.FOV and 0 < dist <= self.MAX_RANGE]
        out.sort(key=lambda p: p[3])
        return out or None


def install(world, counts, miss_every=0, vanish_after=0):
    """drive 모듈의 하드웨어·시계·서버를 가짜로 갈아끼운다."""
    state = {"n": 0, "hits": 0}
    it = iter(counts)

    class Cam:
        calibration_matrix = "fake"
        def get_frame(self):
            # str 을 주면 send_image_and_get_count 가 "파일 경로" 분기로 빠진다
            return types.SimpleNamespace(shape=(480, 640, 3))
        def detect_aruco(self, frame, marker_size=0.02):
            state["n"] += 1
            # 실제 카메라는 흔들림·역광으로 프레임을 흘린다
            if miss_every and state["n"] % miss_every == 0:
                return frame, None
            vis = world.visible()
            if vis:
                state["hits"] += 1
            # 마커가 도중에 아예 사라지는 상황(가림·사각·거리 이탈)
            if vanish_after and state["hits"] > vanish_after:
                return frame, None
            return frame, vis
        def close(self): pass

    class Motor:
        def move(self, l, r): self.cmd = (l, r)
        def close(self): pass

    class LCD:
        shown = None
        def img_show(self, img): LCD.shown = img
        def close(self): pass

    class Resp:
        status_code = 200
        def json(self):
            return {"detected_count": next(it, 0), "saved_filename": "sim.jpg"}

    drive.pinky_cam, drive.pinky_motor, drive.pinky_lcd = Cam(), Motor(), LCD()
    drive.SERVER_URL = "http://sim/predict"
    drive.time = types.SimpleNamespace(sleep=CLOCK.sleep, time=CLOCK.time)
    drive.requests = types.SimpleNamespace(post=lambda *a, **k: Resp())
    drive.cv2 = types.SimpleNamespace(
        imencode=lambda ext, img: (True, types.SimpleNamespace(tobytes=lambda: b"x")))

    # 월드에 반영되는 모터 동작
    def fwd(t, speed=None):
        world.forward(World.CM_PER_SEC * t); CLOCK.sleep(t)
    def back(t, speed=None):
        world.forward(-World.CM_PER_SEC * t); CLOCK.sleep(t)
    def _deg(t, speed):
        scale = 1.0 if speed is None else speed / drive.MOTOR_SPEED
        return World.DEG_PER_SEC * t * scale
    def right(t, speed=None):
        world.turn(_deg(t, speed)); CLOCK.sleep(t)
    def left(t, speed=None):
        world.turn(-_deg(t, speed)); CLOCK.sleep(t)

    # IMU 닫힌 회전. 실제로는 IMU 를 읽으며 도는데, 여기서는 명령 각도만큼
    # (약간의 오차를 섞어) 돌았다고 본다. 닫혀 있으니 시간 오차는 안 쌓인다.
    def _turn_cost(deg):
        """drive.py 의 turn_deg 가 실제로 쓰는 시간. 상수는 drive.py 에서 읽는다.

        예전엔 |deg|/212 + 0.2 로 뭉갰는데, 실제로는 속도 clamp 때문에 더 느리고
        coast 후 정지 대기(0.35초)와 오차 보정 회전이 붙는다. 랩타임을 줄이려면
        이 모델이 맞아야 한다 — 8번 도는 코스라 회전이 제일 큰 비용이다.
        """
        speed = max(drive.TURN_MIN_SPEED,
                    min(drive.TURN_MAX_SPEED,
                        int(abs(deg) * drive.TURN_COAST_RATIO / drive.TURN_COAST_PER_SPEED)))
        rate = World.DEG_PER_SEC * speed / drive.MOTOR_SPEED
        target = max(0.0, abs(deg) - drive.TURN_COAST_PER_SPEED * speed)
        return target / rate + drive.TURN_SETTLE_BASE + drive.TURN_SETTLE_PER_SPEED * speed

    def turn_deg(deg):
        actual = deg * World.TURN_SLIP
        world.turn(actual)
        CLOCK.sleep(_turn_cost(deg))
        err = deg - actual
        if abs(err) > drive.TURN_TOL_DEG:          # drive.py 는 여기서 한 번 더 돈다
            CLOCK.sleep(_turn_cost(err))
        return actual

    drive.move_forward, drive.move_backward = fwd, back
    drive.move_right, drive.move_left = right, left
    drive.turn_deg = turn_deg
    drive.turn_left_deg = lambda d: turn_deg(-abs(d))
    drive.turn_right_deg = lambda d: turn_deg(abs(d))
    drive.read_yaw = lambda: 0.0
    # 도로 유지 주행은 pinky_motor 를 직접 몰아서 이 월드 모델에 안 맞는다.
    # 여기서는 마커 상태기계만 본다. 도로 유지는 tools/road.py 가 실제 사진으로 검증한다.
    drive.move_forward_on_road = lambda t, step=0.25, motor_speed=None: fwd(t)
    # STOP 표지판 검출은 실제 사진이 필요하다. 여기서는 마커 상태기계만 보므로
    # "잠깐 전진하다 멈춘다"로 대신한다. 검출 자체는 tools/stopsign.py 가 검증한다.
    drive.drive_until_stop_sign = lambda max_seconds=8.0, step=0.2: (fwd(0.5), True)[1]
    return LCD


def run(errors, dists, counts, decoys=None, miss_every=0, vanish_after=0, verbose=False):
    """코스를 순서대로 시뮬레이션한다.

    마커를 고정 배치하지 않는다. 실제 코스에서는 직전 마커의 after_track_list 동작이
    로봇을 다음 마커 쪽으로 '대충' 돌려놓기 때문이다. i번째를 처리한 직후에 i+1번째를
    로봇 정면에서 errors[i+1] 도 어긋난 자리에 놓는다. 그 오차를 find_aruco 가
    회수하는지가 검증 대상이다.
    """
    world = World()
    LCD = install(world, counts, miss_every, vanish_after)
    drive.total_apple_count = 0
    decoys = decoys or {}

    def place(i):
        if i >= len(drive.target_list):
            world.markers = {}
            return
        mid = drive.target_list[i]["id"]
        # 같은 마커를 연속으로 쓰는 경우(횡단보도 STOP -> END)는 재배치하지 않는다
        if i > 0 and drive.target_list[i - 1]["id"] == mid and mid in world.markers:
            return
        world.markers = {mid: [errors[i], dists[i]]}
        if i in decoys:
            did, daz = decoys[i]
            world.markers[did] = [daz, dists[i] * 0.8]

    orig = drive.after_target_do_list
    def after_and_advance(index):
        orig(index)
        place(index + 1)
    drive.after_target_do_list = after_and_advance

    place(0)
    silent, real = open(os.devnull, "w"), sys.stdout
    if not verbose:
        sys.stdout = silent
    try:
        drive.run_course()
    finally:
        sys.stdout = real
        drive.after_target_do_list = orig
    return LCD


def check_undefined_names():
    """drive.py 에 정의되지 않은 전역 이름이 있는지 정적으로 본다.

    노트북을 스크립트로 옮기면서 import 를 빠뜨린 적이 있는데, 그 코드 경로를
    시뮬이 안 타면 통과해버린다. 실행 전에 이름부터 확인한다.
    """
    import ast, builtins
    tree = ast.parse(open(drive.__file__).read())
    defined = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            defined.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            defined.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.Global):
            defined.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            defined.update(x.arg for x in a.args + a.kwonlyargs + a.posonlyargs)
            if a.vararg:
                defined.add(a.vararg.arg)
            if a.kwarg:
                defined.add(a.kwarg.arg)
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(used - defined)
    assert not missing, f"drive.py 에 정의 안 된 이름: {missing}"
    print("이름 검사 OK (import 누락 없음)")


def demo():
    check_undefined_names()

    N = len(drive.target_list)
    # 과수원 3곳 x APPLE_CHECK_COUNT 장. 각 묶음에 일부러 튀는 값을 하나씩 섞었다
    # (오탐 1장, 누락 1장). drive.py 는 중앙값을 쓰므로 여기에 흔들리면 안 된다.
    K = drive.APPLE_CHECK_COUNT
    GROUPS = [[1, 1, 2, 1, 1],      # 오탐 1장 -> 정답 1
              [3, 3, 3, 4, 3],      # 오탐 1장 -> 정답 3
              [2, 2, 2, 2, 0]][:3]  # 누락 1장 -> 정답 2
    GROUPS = [(g * K)[:K] if len(g) < K else g[:K] for g in GROUPS]
    COUNTS = [c for g in GROUPS for c in g]
    EXPECT = sum(sorted(g)[len(g) // 2] for g in GROUPS)
    ERR = [10, -15, 20, -10, 15, -20, 10, -15, 5, 0][:N]
    # map.png 실측 구간거리(cm) + 마커는 정지점보다 pose[1] 만큼 더 앞
    DIST = [t["cm"] + t["pose"][1] for t in drive.target_list]
    assert sum(1 for a in drive.after_track_list
               for act, _ in a["actions"] if act == drive.APPLE_COUNT_ACTION) == 3, \
        "사과를 세는 지점이 3곳(과수원 A·B·C)이 아니다"
    assert len(ERR) == N and len(DIST) == N

    CLOCK.t = 0.0
    LCD = run(ERR, DIST, list(COUNTS))
    assert drive.total_apple_count == EXPECT, f"사과 {drive.total_apple_count} (기대 {EXPECT})"
    assert LCD.shown is not None, "LCD 에 아무것도 안 그려졌다"
    assert LCD.shown.size == (320, 240), f"LCD 이미지 크기가 이상하다: {LCD.shown.size}"
    print(f"OK  정상 주행 · 사과 {drive.total_apple_count}개 · 가상 {CLOCK.time():.0f}초")

    CLOCK.t = 0.0
    big = [45, -50, 55, -45, 50, -55, 45, -50, 40, 0][:N]
    run(big, DIST, list(COUNTS))
    assert drive.total_apple_count == EXPECT, f"회전 어긋남에서 사과 {drive.total_apple_count}"
    print(f"OK  회전 ±50도 어긋나도 회수 · 사과 {drive.total_apple_count}개 · 가상 {CLOCK.time():.0f}초")

    # 멀리 있는 마커를 목표 id 로 잘못 읽는 상황. drive.py 는 Z_ACCEPT_MAX 밖을
    # 무시해야 하고, 무시한 뒤에도 진짜 마커를 찾아 완주해야 한다.
    CLOCK.t = 0.0
    world = World()
    LCD = install(world, list(COUNTS))
    drive.total_apple_count = 0
    far = drive.Z_ACCEPT_MAX + 40
    seen = {"n": 0}
    real_detect = world.visible
    def ghost():
        # 앞의 몇 프레임은 "목표 id 가 아주 멀리 보인다" 고 거짓말한다
        seen["n"] += 1
        if seen["n"] <= 6:
            return [[drive.target_list[0]["id"], 0.0, 0.0, far]]
        return real_detect()
    world.visible = ghost
    world.markers = {drive.target_list[0]["id"]: [0, DIST[0]]}
    ok, _ = drive.find_aruco(drive.target_list[0]["id"], drive.RIGHT, drive.SEARCH_COUNT)
    assert ok, "먼 오인식을 걸러낸 뒤 진짜 마커를 못 찾았다"
    assert seen["n"] > 6, "먼 오인식을 그대로 받아들였다"
    print("OK  멀리 있는 마커 오인식을 무시하고 진짜를 찾는다")

    CLOCK.t = 0.0
    run(ERR, DIST, list(COUNTS), decoys={0: (7, 5), 1: (8, -5), 2: (7, 5)})
    assert drive.total_apple_count == EXPECT, f"방해 마커에서 사과 {drive.total_apple_count}"
    print(f"OK  방해 마커가 더 가까이 있어도 목표 선별 · 사과 {drive.total_apple_count}개")

    CLOCK.t = 0.0
    run([0] * N, [9999] * N, [])          # 전부 인식 범위 밖
    assert CLOCK.time() < N * (drive.MARKER_TIMEOUT + 20), \
        f"마커 전무에서 {CLOCK.time():.0f}초 — 타임아웃이 안 먹는다"
    assert drive.total_apple_count == 0
    print(f"OK  마커 전무에서도 무한루프 없이 종료 (가상 {CLOCK.time():.0f}초)")

    CLOCK.t = 0.0
    run(ERR, DIST, list(COUNTS), miss_every=3)
    assert drive.total_apple_count == EXPECT, f"간헐 인식 실패에서 사과 {drive.total_apple_count}"
    print(f"OK  3프레임마다 인식 실패해도 완주 · 가상 {CLOCK.time():.0f}초")

    CLOCK.t = 0.0
    run(ERR, DIST, list(COUNTS), vanish_after=3)
    assert CLOCK.time() < N * (drive.MARKER_TIMEOUT + 20), \
        f"마커 소실에서 {CLOCK.time():.0f}초 — 무한루프"
    print(f"OK  주행 중 마커 소실에도 멈추지 않고 종료 (가상 {CLOCK.time():.0f}초)")

    # 각도 정렬 루프 단독 검증.
    # 배포 원본은 "탐색은 성공했는데 정렬 직후 마커가 영영 사라진" 상황에서 2000회를
    # 돌려도 안 끝났다(무한 우회전). 반드시 유한 시간에 끝나야 한다.
    CLOCK.t = 0.0
    world = World()
    install(world, [])
    turns = {"n": 0}
    def counting(t, speed=None):
        turns["n"] += 1
        CLOCK.sleep(t)
        assert turns["n"] < 2000, "각도 정렬 루프가 끝나지 않는다 (무한 우회전)"
    drive.move_left = drive.move_right = drive.move_forward = counting
    drive.find_aruco = lambda *a: (True, [[1, 0, 0, 100]])
    drive.detect_target_aruco = lambda i: (False, None)
    silent, real = open(os.devnull, "w"), sys.stdout
    sys.stdout = silent
    try:
        result = drive.track_target_aruco_marker(1, [0, 55, drive.RIGHT], drive.SEARCH_COUNT)
    finally:
        sys.stdout = real
    assert result is False, f"마커가 영영 안 보이는데 성공을 반환했다: {result}"
    print(f"OK  정렬 중 마커 영구 소실 → {CLOCK.time():.0f}초 만에 포기 (회전 {turns['n']}회)")


if __name__ == "__main__":
    demo()
