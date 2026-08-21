"""대회 도면(26wcrc_final.pdf 3페이지)에서 주행 경로를 뽑는다.

    python tools/map.py            # PATH_LEGS 출력 + 검증
    python tools/map.py --plot     # 도로·경로 그림 (SP 환경변수 폴더에)

도면을 200dpi 로 렌더하면 1픽셀 = 1mm 다 (경기장 2100x1100mm).
도로(흰색)만 남기고 -> 세선화 -> 지점 사이 최단경로 -> 전체를 하나로 이어 붙이고
-> 도로 안에 머무는 조건으로 부드럽게 편 뒤 -> 구간별 (누적cm, 상대헤딩) 으로 자른다.

왜 이어 붙여서 펴는가:
  세선화 경로를 구간마다 따로 쓰면 두 군데서 망가진다. ① 8방향 픽셀이라 방향이
  0/45도로 튄다. ② 교차로에서 세선이 갈래 쪽으로 휘어 도로에 없는 급꺾임이 생긴다
  (실측: 구간 마지막 3cm 에서 헤딩이 +28 -> +112 로 튀었다). 헤딩은 상대값이라
  구간 끝이 틀리면 다음 구간이 통째로 돌아간다. 전체를 잇고 펴면 둘 다 사라진다.

★ 이 파일이 drive.py PATH_LEGS 의 원본이다. 손으로 고치지 말고 여기서 다시 뽑을 것.
"""
import heapq
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "26wcrc_final.pdf")
SP = os.environ.get("SP", "/tmp")

# 도면 위 지점(픽셀=mm). 경기장만 잘라낸 좌표계 기준.
PTS = {"START": (89, 551), "교차로1": (483, 420), "과수원A": (473, 798),
       "교차로2": (924, 441), "과수원B": (945, 220), "교차로3": (1050, 735),
       "과수원C": (1082, 977), "교차로4": (1523, 725), "하차장": (1533, 299),
       "END": (2032, 546)}

# 주행 순서. 과수원·하차장은 막다른 가지라 마커 도착 후 동작에서 왕복한다.
ROUTE = ["START", "교차로1", "교차로2", "교차로3", "교차로4", "END"]

# 관통 도로에 붙은 막다른 가지. 세선화 전에 잘라낸다.
# 이게 있으면 교차로에서 세선이 가지 쪽으로 휘어 도로에 없는 급꺾임이 생기고
# (실측: 구간 끝 헤딩이 +118도로 튄다), 헤딩은 상대값이라 다음 구간이 통째로 돈다.
# 잘라내면 교차로가 아예 없어지고 매끈한 S자 하나만 남는다.
BRANCHES = [("교차로1", "과수원A"), ("교차로2", "과수원B"),
            ("교차로3", "과수원C"), ("교차로4", "하차장")]
BRANCH_KEEP_MM = 300      # 교차로 중심에서 이만큼 떨어진 데서부터 자른다
BRANCH_HALF_MM = 100      # 자르는 폭의 절반. 가지 반폭(77mm)보다 넉넉히
MAX_DEG_PER_CM = 12       # 로봇이 실제로 돌 수 있는 곡률 상한 (도/cm)
                          # 16 이면 급커브에서 못 따라가 헤딩이 3.6도 모자란다
TAIL_CM = 2.0             # 구간 양끝 이만큼은 헤딩을 붙들고 곧게 드나든다
                          # (스윕 결과: 2cm 에서 바퀴 여유 +8.9mm, 곡률 최소)

FIELD = (279, 342, 2382, 1401)                    # 도면에서 경기장만 잘라낼 사각형
WHITE_LO, WHITE_HI = (0, 0, 190), (179, 45, 255)  # 도로 = 흰색
WHEEL_HALF_MM = 55.5      # 바퀴 반폭. 도로 가장자리까지 이만큼은 남아야 한다
STEP_CM = 1.0             # 웨이포인트 간격
HEAD_MM = 20              # 헤딩을 재는 앞뒤 창
JUNCTION_MM = 0           # 교차로 반경. 이 안의 헤딩은 안 믿고 이어 붙인다


def road_mask():
    out = os.path.join(SP, "map")
    if not os.path.exists(out + "-3.png"):
        subprocess.run(["pdftoppm", "-r", "200", "-f", "3", "-l", "3", "-png",
                        PDF, out], check=True)
    page = cv2.imread(out + "-3.png")
    assert page is not None, "도면 렌더 실패"
    x0, y0, x1, y1 = FIELD
    m = cv2.inRange(cv2.cvtColor(page[y0:y1, x0:x1], cv2.COLOR_BGR2HSV),
                    np.array(WHITE_LO), np.array(WHITE_HI))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # 도로는 하나로 이어져 있다. 제일 큰 덩어리만 남기면 글자·범례가 떨어져 나간다.
    _, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(lab == big, 255, 0).astype(np.uint8)

    # 도면에는 횡단보도 줄무늬·장애물이 도로 위에 검게 그려져 있다. 그대로 두면
    # 마스크에 구멍이 나고 세선이 그걸 피해 돌아가 도로에 없는 스웨이브가 생긴다
    # (실측: leg4 가 이것 때문에 30mm 밀렸다). 바깥에서 물을 채워 구멍만 메운다.
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(m), 8)
    for k in range(1, n):
        x, y, w2, h2, area = st[k]
        touches = x == 0 or y == 0 or x + w2 >= m.shape[1] or y + h2 >= m.shape[0]
        # 도로 밖 잔디는 S자 때문에 위아래로 갈려 있다 — 가장자리에 닿거나 큰 덩어리는
        # 잔디지 구멍이 아니다. 줄무늬 하나는 아무리 커도 도로 폭(154mm)을 못 넘는다.
        if not touches and area < 154 * 154:
            m[lab == k] = 255

    for j, tip in BRANCHES:
        a, b = np.array(PTS[j], float), np.array(PTS[tip], float)
        u = (b - a) / np.hypot(*(b - a))
        v = np.array([-u[1], u[0]])
        # 굵은 선으로 자르면 끝이 둥글어서 교차로 쪽으로 파고든다. 가지 축 방향으로만
        # 정확히 시작하는 사각형으로 자른다 — 관통 도로는 한 픽셀도 안 건드린다.
        p0, p1 = a + u * BRANCH_KEEP_MM, b + u * 200
        quad = np.array([p0 + v * BRANCH_HALF_MM, p1 + v * BRANCH_HALF_MM,
                         p1 - v * BRANCH_HALF_MM, p0 - v * BRANCH_HALF_MM])
        cv2.fillPoly(m, [quad.astype(np.int32)], 0)
    _, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == big, 255, 0).astype(np.uint8)


NB = [(dx, dy, (dx * dx + dy * dy) ** 0.5)
      for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def shortest(node, xs, ys, a, b):
    """세선화된 도로 위 두 지점의 최단경로(픽셀=mm)."""
    def snap(p):
        i = int(np.hypot(xs - p[0], ys - p[1]).argmin())
        return (int(xs[i]), int(ys[i]))

    a, b = snap(a), snap(b)
    dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == b:
            break
        if d > dist.get(u, 1e18):
            continue
        for dx, dy, w in NB:
            v = (u[0] + dx, u[1] + dy)
            if v in node and d + w < dist.get(v, 1e18):
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (d + w, v))
    assert b in dist, "경로 없음 — PTS 좌표가 도로 밖이다"
    out, u = [b], b
    while u != a:
        u = prev[u]
        out.append(u)
    return np.array(out[::-1], float)


def relax(px, dist, rounds=60, alpha=0.5, keep=WHEEL_HALF_MM + 8):
    """도로 안에 머무는 조건으로 경로를 편다.

    이웃 평균 쪽으로 조금씩 당기되, 당긴 자리의 도로 여유가 keep 아래로 떨어지면
    그 점만 되돌린다. 그래서 곧은 데서는 곧아지고 좁은 커브에서는 안 밀린다.
    양끝은 고정한다 (START / END 위치는 정해져 있다).
    """
    p = px.copy()
    h, w = dist.shape
    for _ in range(rounds):
        mid = (p[:-2] + p[2:]) / 2
        cand = p[1:-1] + alpha * (mid - p[1:-1])
        ij = np.clip(np.round(cand).astype(int), [0, 0], [w - 1, h - 1])
        ok = dist[ij[:, 1], ij[:, 0]] >= keep
        p[1:-1] = np.where(ok[:, None], cand, p[1:-1])
    return p


def arclen(px):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(px, axis=0).T))])


def headings(px, junctions=()):
    """경로 위 모든 점의 접선 방향(도). 화면 좌표라 오른쪽으로 도는 것이 + 다.

    ★ 전체 경로에서 한 번에 잰다. 구간마다 따로 재면 구간 경계에서 앞뒤 창이
    한쪽으로 잘려 그 점 헤딩이 잡음이 된다 (실측: 구간 끝이 +135도로 튀었다).
    헤딩은 상대값이라 구간 끝이 틀리면 다음 구간이 통째로 돌아간다.

    junctions 근처는 세선이 가지 쪽으로 휘어 도로에 없는 V자 꺾임을 만든다
    (실측 -55도, +36도). 실제 도로는 교차로를 곧게 통과한다. 그 구간은 재지 않고
    양옆 성한 값을 이어 붙인다.
    """
    n = len(px)
    i = np.arange(n)
    a = px[np.maximum(i - HEAD_MM, 0)]
    b = px[np.minimum(i + HEAD_MM, n - 1)]
    hd = np.unwrap(np.degrees(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])), period=360)
    bad = np.zeros(n, bool)
    for j in junctions:
        bad[max(0, j - JUNCTION_MM):min(n, j + JUNCTION_MM)] = True
    if bad.any() and (~bad).any():
        hd[bad] = np.interp(i[bad], i[~bad], hd[~bad])
    return hd


def profile(px, hd, step_cm=STEP_CM):
    """경로(mm) + 그 위의 헤딩 -> [(누적 cm, 상대 헤딩 도)].

    상대로 두는 이유: 구간마다 교차로 회전이 끼어들어도 그 회전이 통째로 흡수된다.
    절대 방위로 두면 회전 한 번 어긋날 때마다 경로 전체가 틀어진다.
    """
    s = arclen(px)
    total = s[-1] / 10.0
    idx = lambda d: int(np.clip(np.searchsorted(s, d * 10), 0, len(px) - 1))
    # 구간 양끝은 교차로다. 가지를 잘라내도 그루터기가 남아 세선이 그쪽으로 휘고,
    # 양끝 몇 cm 에서 헤딩이 ±100도까지 튄다 (실측). 실제 도로는 교차로를 곧게
    # 통과하고 회전은 마커 도착 후 동작이 한다. 양끝은 성한 헤딩으로 붙든다.
    lo, hi = min(TAIL_CM, total / 2), max(total - TAIL_CM, total / 2)
    h0 = hd[idx(lo)]
    ds = list(np.arange(0.0, total, step_cm)) + [total]
    raw = [(hd[idx(min(max(d, lo), hi))] - h0 + 180) % 360 - 180 for d in ds]

    # 로봇이 못 도는 곡률은 프로파일에 적어봐야 못 따라간다 (실측: 26도/cm 구간에서
    # 5도 모자랐다). 남은 급꺾임은 어차피 교차로 잔재이므로 여기서 눌러 담는다.
    out = [raw[0]]
    for d0, d1, h in zip(ds, ds[1:], raw[1:]):
        lim = MAX_DEG_PER_CM * (d1 - d0)
        out.append(out[-1] + max(-lim, min(lim, h - out[-1])))
    return [(round(d, 1), round(h, 1)) for d, h in zip(ds, out)], total, h0


def reconstruct(pts):
    """(누적cm, 헤딩) 만 보고 달렸을 때의 궤적(mm). 로봇이 실제로 보는 정보다."""
    cs = np.array([c for c, _ in pts], float)
    hs = np.radians([h for _, h in pts])
    d = np.diff(cs)
    mid = (hs[:-1] + hs[1:]) / 2
    return np.vstack([[0, 0],
                      np.cumsum(np.stack([d * np.cos(mid), d * np.sin(mid)], 1), 0)]) * 10


def to_image(pts, h0, origin):
    c, sn = np.cos(np.radians(h0)), np.sin(np.radians(h0))
    return reconstruct(pts) @ np.array([[c, sn], [-sn, c]]) + origin


def main():
    mask = road_mask()
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sk = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    ys, xs = np.nonzero(sk)
    node = {(int(x), int(y)) for x, y in zip(xs, ys)}
    print(f"도로 {mask.shape[1]}x{mask.shape[0]}px(=mm)  세선 {len(node)}px  "
          f"도로폭 중앙값 {np.median(dist[sk > 0]) * 2:.0f}mm (실측 154mm)")

    # 교차로마다 끊어 푼다. START->END 를 한 번에 풀면 교차로 안쪽으로 코너를 잘라
    # 중심에서 최대 10cm 비껴간다 — 거기서 과수원으로 90도 꺾어야 하므로 못 쓴다.
    route, cuts = [], [0]
    for a, b in zip(ROUTE, ROUTE[1:]):
        seg = shortest(node, xs, ys, PTS[a], PTS[b])
        route.append(seg if not route else seg[1:])
        cuts.append(sum(len(r) for r in route) - 1)
    route = np.vstack(route)
    raw_len = arclen(route)[-1] / 10
    route = relax(route, dist)
    print(f"전체 경로 {raw_len:.1f}cm -> 편 뒤 {arclen(route)[-1] / 10:.1f}cm "
          f"(8방향 픽셀 계단이 6% 부풀린 것을 편 결과)\n")

    hd_all = headings(route, cuts[1:-1])
    legs = []
    for i, (a, b) in enumerate(zip(ROUTE, ROUTE[1:])):
        px = route[cuts[i]:cuts[i + 1] + 1]
        pts, total, h0 = profile(px, hd_all[cuts[i]:cuts[i + 1] + 1])
        legs.append((a, b, pts, total, h0, px))
        print(f"{a:>6} -> {b:<6} {total:5.1f}cm  총 {pts[-1][1]:+6.1f}도  "
              f"웨이포인트 {len(pts)}개")

    # 검증 ①: START->교차로1. 실주행에서 잰 START->마커1 이 47cm 이고 마커는
    # 교차로보다 2~3cm 뒤에 서 있으므로 44~45cm 가 나와야 한다.
    # (생 세선 경로는 47.5cm 로 나오는데, 8방향 픽셀 계단 때문에 6% 부풀려진 값이다)
    assert abs(legs[0][3] - 44.5) < 2, f"START->교차로1 이 {legs[0][3]:.1f}cm (기대 44.5)"

    # 검증 ②: 로봇이 실제로 보는 정보((누적cm, 헤딩))만으로 달렸을 때 바퀴가
    # 도로 위에 있나. 이것만 지키면 된다. 도면 끝단은 도로가 거기서 끊겨 여유가
    # 작으므로, 절대값이 아니라 "도면 경로 자신보다 나쁘지 않은가" 로 본다.
    print()
    worst = 1e9
    for a, b, pts, total, h0, px in legs:
        ours = to_image(pts, h0, px[0])
        ij = np.clip(np.round(ours).astype(int),
                     [0, 0], [mask.shape[1] - 1, mask.shape[0] - 1])
        mine = dist[ij[:, 1], ij[:, 0]]
        idx = np.clip(np.searchsorted(arclen(px), [c * 10 for c, _ in pts]),
                      0, len(px) - 1)
        ideal = np.array([dist[int(round(y)), int(round(x))] for x, y in px[idx]])
        # START/END 는 도로가 거기서 끊겨 도면 경로 자신도 여유가 없다
        m = (mine - np.minimum(ideal, WHEEL_HALF_MM))[3:-3].min()
        worst = min(worst, m)
        print(f"{a:>6} -> {b:<6} 바퀴 여유 {m:+6.1f}mm (도면 경로 대비)")
    # 도면 중심선 자체가 최적이므로 그보다 조금 나쁜 건 허용한다. 실제 여유는
    # 도로 반폭 77mm - 바퀴 반폭 55.5mm = 21.5mm 이고, 그 절반까지는 쓸 수 있다.
    print(f"\n최소 {worst:+.1f}mm — 도면 중심선 대비. 실제 여유 21.5mm 중 "
          f"{21.5 + min(0, worst):.1f}mm 남는다")
    assert worst >= -10, "경로대로 가면 바퀴가 도로를 나간다"

    print("\n" + "=" * 72)
    print("PATH_LEGS = [")
    for i, (a, b, pts, total, _, _) in enumerate(legs):
        print(f"    # [{i}] {a} -> {b}   도면 {total:.0f}cm, 총 {pts[-1][1]:+.0f}도")
        print("    [" + ", ".join(f"({c:g}, {h:+g})" for c, h in pts) + "],")
    print("]")

    if "--plot" in sys.argv:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        for i, (_, _, pts, _, h0, px) in enumerate(legs):
            for q in px.astype(int):
                cv2.circle(vis, tuple(q), 2, (90, 90, 90), -1)
            for q in to_image(pts, h0, px[0]).astype(int):
                cv2.circle(vis, tuple(q), 4, cols[i % len(cols)], -1)
        for name, (x, y) in PTS.items():
            cv2.circle(vis, (x, y), 12, (255, 255, 255), 2)
        cv2.imwrite(f"{SP}/map_paths.png", vis)
        print(f"\n그림: {SP}/map_paths.png  (회색=도면 경로, 색=프로파일대로 달린 궤적)")


if __name__ == "__main__":
    main()
