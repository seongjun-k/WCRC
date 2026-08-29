"""주행이 끝난 뒤 그 사진으로 전처리 값을 다시 고른다. 대회장에서 오프라인으로 돈다.

    .venv/bin/python tools/tune.py 1 2 1        # 과수원 A/B/C 의 실제 빨간 사과 수
    .venv/bin/python tools/tune.py              # 정답 없이, 지금 값으로 몇 개 세는지만 본다
    .venv/bin/python tools/tune.py 1 2 1 --keep # 로봇에서 사진 안 받아오고 있는 걸로

로봇에서 사진을 받아 아래 후보를 전부 돌려보고, 정답을 맞히면서 여유가 가장 큰
조합을 tools/tune.json 에 쓴다. serve.py 가 기동할 때 읽는다 (wrun 이 알아서 재기동).
"""
import json
import os
import subprocess
import sys

# 인터넷 없이 돌리기 위해 필요하다 (ultralytics 가 import 할 때 DNS 를 친다).
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHOTS = os.path.join(HERE, "tune_shots")
ROBOT = "pinky@192.168.4.1"

# ---------------------------------------------------------------- 훑을 후보
# 시간이 모자라면 여기를 줄인다 (조합 수 = 곱). 지금 6x3x2x3 = 108.
GRID_CLIP = [None, 1.5, 2.0, 3.0, 4.0, 6.0]
GRID_IOU = [0.5, 0.6, 0.7]
GRID_SCALE = [2, 3]
GRID_CONF = [0.15, 0.25, 0.35]   # app.py 가 0.25 를 주지만 serve.py 가 덮어쓴다
RATIO = 0.75


def pull():
    os.makedirs(SHOTS, exist_ok=True)
    for f in os.listdir(SHOTS):
        os.remove(os.path.join(SHOTS, f))
    # 주행마다 폴더가 하나씩 생긴다. 이름이 날짜시간이라 사전순 = 시간순이다.
    q = subprocess.run(["ssh", ROBOT, "ls -d ~/apple_shots/*/ 2>/dev/null | tail -1"],
                       capture_output=True, text=True)
    last = q.stdout.strip()
    if not last:
        sys.exit("로봇에 주행 사진 폴더가 없다")
    print(f"가져오는 폴더: {last}")
    r = subprocess.run(["scp", "-q", f"{ROBOT}:{last}*.jpg", SHOTS],
                       capture_output=True)
    if r.returncode:
        sys.exit(f"사진을 못 받아왔다: {r.stderr.decode().strip()}")


def clahe(img, clip):
    if not clip:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clip, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load(paths, scale):
    """(크롭 + 확대) 까지는 clip 과 무관하므로 조합마다 다시 안 만든다."""
    out = {}
    for p in paths:
        im = cv2.imread(p)
        top = im[:int(im.shape[0] * RATIO)]
        out[p] = cv2.resize(top, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
    return out


def median(xs):
    return sorted(xs)[len(xs) // 2] if xs else 0


def main(argv):
    truth = [int(a) for a in argv if a.isdigit()]
    if "--keep" not in argv:
        pull()
    paths = sorted(os.path.join(SHOTS, f) for f in os.listdir(SHOTS)
                   if f.endswith(".jpg"))
    if not paths:
        sys.exit(f"{SHOTS} 에 사진이 없다")

    # apple_2_0.jpg -> 과수원 2. 주행 한 번에 과수원마다 여러 장이 찍힌다.
    groups = {}
    for p in paths:
        groups.setdefault(os.path.basename(p).split("_")[1], []).append(p)
    keys = sorted(groups)
    print(f"사진 {len(paths)}장 / 과수원 {len(keys)}곳 {keys}")
    if truth and len(truth) != len(keys):
        sys.exit(f"정답을 {len(keys)}개 줘야 한다 (준 건 {len(truth)}개)")

    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "runs", "detect", "train", "weights", "best.pt"))

    rows = []
    for scale in GRID_SCALE:
        cache = load(paths, scale)
        for clip in GRID_CLIP:
            prepped = {p: clahe(im, clip) for p, im in cache.items()}
            for iou in GRID_IOU:
              for conf in GRID_CONF:
                per = {}
                margin = 1.0
                for k in keys:
                    cs = []
                    for p in groups[k]:
                        b = model(prepped[p], conf=conf, iou=iou, verbose=False)[0].boxes
                        cs.append(len(b))
                        for x in b:
                            margin = min(margin, float(x.conf) - conf)
                    per[k] = median(cs)
                hit = sum(per[k] == t for k, t in zip(keys, truth)) if truth else -1
                rows.append((hit, margin, scale, clip, iou, conf, per))
                print(f"  x{scale} clip={str(clip):<4} iou={iou} conf={conf}  "
                      f"셈 {[per[k] for k in keys]}"
                      + (f"  맞음 {hit}/{len(keys)}" if truth else "")
                      + f"  여유 {margin:+.2f}")

    if not truth:
        print("\n정답을 안 줬으니 고르지 않는다. 실제 개수를 인자로 주면 그때 고른다.")
        return 0

    # 1순위 정답 개수, 2순위 여유. 0.26 으로 겨우 넘긴 설정은 조명이 바뀌면 무너진다.
    best = max(rows, key=lambda r: (r[0], r[1]))
    hit, margin, scale, clip, iou, conf, per = best
    print(f"\n고른 값: 확대 x{scale} / CLAHE {clip} / NMS iou {iou} / conf {conf}")
    print(f"  {hit}/{len(keys)} 곳 정답, 문턱까지 여유 {margin:+.2f}")
    if hit < len(keys):
        print("  !! 전부 맞히는 조합이 없다. 사진을 직접 열어볼 것 "
              f"({SHOTS})")

    cfg = {"scale": scale, "clip": clip, "iou": iou, "conf": conf, "ratio": RATIO}
    with open(os.path.join(HERE, "tune.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\ntools/tune.json 에 썼다. 서버를 다시 띄우면 적용된다:")
    print("  tools/go.sh 를 다시 실행하거나, 서버만 재시작")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
