"""빨간 사과를 색으로 찾아 YOLO 라벨을 만든다.

    python tools/redlabel.py raw            # raw/images -> raw/labels
    python tools/redlabel.py raw --check          # 전체를 그려서 눈으로 확인
    python tools/redlabel.py raw "orcha_*" --check # 일부만
    python tools/redlabel.py raw --force    # 이미 있는 라벨도 덮어쓴다 (기본은 안 덮는다)

기본으로 덮어쓰지 않는 이유: 손으로 검수한 라벨을 날린 적이 있다. 되돌릴 방법이 없다.

왜 모델 대신 색인가: 부트스트랩용 모델의 확신도가 0.13 이라 자동 라벨로 못 쓴다.
반면 이 사과는 채도 높은 빨간 구(球)라 색+모양만으로 거의 완벽하게 잡힌다.
초록 사과는 규정 3번상 세지 않으므로 일부러 무시한다.

바닥 그림에도 빨간 부분이 있어서 두 가지로 거른다:
  - 크기 (사과는 1000px^2 이상, 바닥 조각은 그 아래)
  - 위치 (나무는 화면 위쪽. 아래 38% 는 도로·바닥이라 버린다)
그래도 완벽하진 않다. 만든 뒤 --check 로 눈으로 훑고, 이상하면 labelImg 로 고친다.
"""
import glob
import os
import sys

import cv2
import numpy as np

AREA_MIN, AREA_MAX = 800, 9000
ASPECT = (0.6, 1.7)          # 사과는 대략 원형
FILL_MIN = 0.55              # 외접 사각형을 채우는 비율 (완전한 원 = 0.785)
Y_TOP = 0.62                 # 이 아래쪽(바닥)은 버린다


def find_apples(im):
    h, w = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # 빨강은 H 가 0 과 179 양쪽 끝에 걸쳐 있어 두 번 뽑아 합친다
    m = (cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255])) |
         cv2.inRange(hsv, np.array([170, 120, 80]), np.array([179, 255, 255])))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if not (AREA_MIN <= a <= AREA_MAX):          continue
        if not (ASPECT[0] < bw / bh < ASPECT[1]):    continue
        if a / (bw * bh) < FILL_MIN:                 continue
        if (y + bh) > Y_TOP * h:                     continue
        out.append((x, y, bw, bh))
    return out


def audit_red(im, amin=120):
    """위치·크기 필터를 거의 풀고 빨간 덩어리를 찾는다.

    라벨이 0인데 여기서 뭔가 나오면 "사과가 있는데 라벨을 안 붙인" 사진이다.
    그런 사진은 모델에게 "이건 사과가 아니다" 라고 가르치므로 오히려 해롭다.
    실제로 자동 촬영한 배경 사진 20장 전부가 여기 걸렸다 (나무가 멀리 작게 찍혀서
    크기 필터를 통과 못 했다).
    """
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, np.array([0, 110, 70]), np.array([12, 255, 255])) |
         cv2.inRange(hsv, np.array([168, 110, 70]), np.array([179, 255, 255])))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    hits = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < amin:                              continue
        if not (0.5 < bw / bh < 2.0):             continue
        if a / (bw * bh) < 0.5:                   continue
        hits.append((x, y, bw, bh, a))
    return hits


def sheets(root="raw", pat="*", per=20):
    """라벨을 그린 contact sheet 를 만들고, 의심스러운 사진을 짚어준다."""
    imgs = sorted(glob.glob(f"{root}/images/{pat}.jpg"))
    assert imgs, f"{root}/images/{pat}.jpg 없음"
    os.makedirs("labelcheck", exist_ok=True)
    for f in glob.glob("labelcheck/*.jpg"):
        os.remove(f)
    suspect, nbox = [], 0
    tiles, page = [], 1
    for f in imgs:
        im = cv2.imread(f)
        h, w = im.shape[:2]
        lp = f"{root}/labels/" + os.path.splitext(os.path.basename(f))[0] + ".txt"
        rows = [r.split() for r in open(lp)] if os.path.exists(lp) else []
        for r in rows:
            cx, cy, bw, bh = (float(v) for v in r[1:5])
            x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
            cv2.rectangle(im, (x1, y1), (x1+int(bw*w), y1+int(bh*h)), (0, 255, 255), 3)
        nbox += len(rows)
        bad = len(rows) == 0 and audit_red(im)
        if bad:
            suspect.append(os.path.basename(f))
            cv2.rectangle(im, (2, 2), (w-3, h-3), (0, 0, 255), 8)
            for x, y, bw, bh, a in audit_red(im):
                cv2.rectangle(im, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        t = cv2.resize(im, (320, 240))
        cv2.putText(t, os.path.basename(f)[-8:-4] + (" !!" if bad else ""),
                    (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(t)
        if len(tiles) == per:
            _save(tiles, page); tiles, page = [], page+1
    if tiles:
        _save(tiles, page)
    print(f"이미지 {len(imgs)}장 / 박스 {nbox}개")
    print(f"contact sheet -> labelcheck/*.jpg  (노란 박스 = 붙은 라벨)")
    if suspect:
        print(f"\n★ 의심 {len(suspect)}장 — 라벨이 없는데 빨간 덩어리가 보인다 (빨간 테두리):")
        for s in suspect[:30]:
            print("   ", s)
        print("   사과가 맞으면 라벨을 붙이거나 그 사진을 지울 것. 그냥 두면 해롭다.")
    else:
        print("\n의심 사진 없음")


def _save(tiles, page):
    while len(tiles) % 5:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    rows = [np.hstack(tiles[i:i+5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(f"labelcheck/page{page:02d}.jpg", np.vstack(rows))


def main(root="raw", check=False, force=False):
    imgs = sorted(glob.glob(f"{root}/images/*.jpg"))
    assert imgs, f"{root}/images 에 jpg 가 없다"
    os.makedirs(f"{root}/labels", exist_ok=True)
    hist, skipped = {}, 0
    for f in imgs:
        im = cv2.imread(f)
        h, w = im.shape[:2]
        lp = f"{root}/labels/" + os.path.splitext(os.path.basename(f))[0] + ".txt"
        if not force and os.path.exists(lp):
            # 손으로 검수한 라벨을 덮어쓰면 되돌릴 방법이 없다. --force 를 줘야 덮는다.
            skipped += 1
            continue
        boxes = find_apples(im)
        hist[len(boxes)] = hist.get(len(boxes), 0) + 1
        with open(lp, "w") as fp:
            for x, y, bw, bh in boxes:
                fp.write(f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")
    print(f"이미지 {len(imgs)}장 / 새로 라벨 {sum(hist.values())}장 / 건너뜀 {skipped}장")
    print(f"사과 개수 분포 {dict(sorted(hist.items()))}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    root = a[0] if a else "raw"
    if "--check" in sys.argv:
        sheets(root, a[1] if len(a) > 1 else "*")
    else:
        main(root, force="--force" in sys.argv)
