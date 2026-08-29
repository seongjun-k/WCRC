"""raw/images 의 새 사진으로 YOLO 를 자동 재학습한다. PC(RTX 4060)에서 실행.

독립 실행 파일이다 — tools/dataset.py 는 건드리지 않는다. wlabel/wsplit/wtrain 과
로직(라벨 -> 분할 -> 학습)은 같지만, 이 파일 하나로 매번 이어서 돈다.

    python tools/auto_train.py            # epochs=80
    python tools/auto_train.py 50         # epochs 바꾸려면 인자로
    wautotrain                            # alias (~/.wcrc_aliases)

실행할 때마다 하는 일:
  1) raw/images 중 아직 라벨 없는 사진을 자동 라벨링
     (현재 실전용 best.pt 가 있으면 그걸로, 없으면 COCO 사전학습 yolov8s.pt 로 부트스트랩)
  2) apple/train, apple/val 로 8:2 분할 (wsplit 과 같은 고정 위치)
  3) YOLO 학습 — runs/detect/train/weights/best.pt 고정 경로도 같이 갱신
  4) 이번 라운드의 학습 결과(가중치·커브·로그)를 runs/auto_train/<날짜시간횟수>/ 에 스냅샷으로 남긴다

주의: 1번은 라벨 "초안"만 만들 뿐 사람 검수(labelImg)를 거치지 않는다. 헛박스 하나가
로봇 점수에 그대로 반영되므로(README 참고), 대회 직전엔 wcheck 로 개수 일치율을
반드시 확인하고 나서 실전에 쓸 것.
"""
import glob
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "apple", "data.yaml")
BEST_PT = os.path.join(ROOT, "runs", "detect", "train", "weights", "best.pt")
YOLOV8S_PT = os.path.join(ROOT, "pt_file", "yolov8s.pt")
AUTO_TRAIN_DIR = os.path.join(ROOT, "runs", "auto_train")


def make_run_dir():
    """실행마다 새 폴더. 이름은 날짜시간 + 그날 몇 번째인지 (drive.py 의 apple_shots 폴더와 같은 규칙)."""
    os.makedirs(AUTO_TRAIN_DIR, exist_ok=True)
    day = time.strftime("%Y%m%d")
    n = sum(1 for d in os.listdir(AUTO_TRAIN_DIR) if d.startswith(day)) + 1
    path = os.path.join(AUTO_TRAIN_DIR, f"{time.strftime('%Y%m%d%H%M')}{n:02d}")
    os.makedirs(path, exist_ok=True)
    return path


def auto_label(src, model_path, conf=0.25):
    """모델로 예측해 YOLO 포맷 라벨 초안 생성. 이미 라벨 있는 이미지는 건너뛴다."""
    from ultralytics import YOLO

    img_dir, lbl_dir = os.path.join(src, "images"), os.path.join(src, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    model = YOLO(model_path)
    keep = {i for i, n in model.names.items() if n == "apple"} or set(model.names)
    print(f"model={model_path} keep={[model.names[i] for i in sorted(keep)]} conf={conf}")

    total, skipped = 0, 0
    for r in model.predict(source=img_dir, conf=conf, stream=True, verbose=False):
        stem = os.path.splitext(os.path.basename(r.path))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            skipped += 1
            continue
        lines = [
            "0 " + " ".join(f"{v:.6f}" for v in box)
            for box, c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist())
            if int(c) in keep
        ]
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))
        total += len(lines)
    print(f"박스 {total}개 생성 (건너뜀 {skipped}장) -> {lbl_dir}")


def split_dataset(src, dst, val_every=5):
    """8:2 분할. 정렬 후 5장마다 1장을 val 로."""
    imgs = sorted(glob.glob(os.path.join(src, "images", "*.jpg")))
    if not imgs:
        sys.exit(f"이미지가 없다: {src}/images")

    for part in ("train", "val"):
        os.makedirs(os.path.join(dst, part, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst, part, "labels"), exist_ok=True)

    n = {"train": 0, "val": 0}
    for i, img in enumerate(imgs):
        part = "val" if i % val_every == 0 else "train"
        stem = os.path.splitext(os.path.basename(img))[0]
        shutil.copy2(img, os.path.join(dst, part, "images", os.path.basename(img)))
        lbl = os.path.join(src, "labels", stem + ".txt")
        open(os.path.join(dst, part, "labels", stem + ".txt"), "w").write(
            open(lbl).read() if os.path.exists(lbl) else ""
        )
        n[part] += 1
    print(f"train {n['train']}장 / val {n['val']}장 -> {dst}")


def train_model(epochs):
    """RTX 4060 Laptop(8GB) 기준 설정. 결과는 BEST_PT 고정 경로에 복사하고, 이번
    라운드 결과 폴더(ultralytics 가 만든 runs/detect/trainN)를 그대로 돌려준다."""
    from ultralytics import YOLO

    base = YOLOV8S_PT if os.path.exists(YOLOV8S_PT) else "yolov8s.pt"
    r = YOLO(base).train(
        data=DATA_YAML,
        epochs=int(epochs),
        imgsz=640,
        batch=16,
        device=0,
        patience=0,
        lr0=0.001,
        freeze=10,
    )

    src = os.path.join(str(r.save_dir), "weights", "best.pt")
    if os.path.abspath(src) != os.path.abspath(BEST_PT):
        os.makedirs(os.path.dirname(BEST_PT), exist_ok=True)
        shutil.copy2(src, BEST_PT)
    print(f"\n{src} -> {BEST_PT}")
    return str(r.save_dir)


def run(epochs=80):
    epochs = int(epochs)
    raw_dir = os.path.join(ROOT, "raw")
    raw_images = os.path.join(raw_dir, "images")
    if not os.path.isdir(raw_images) or not os.listdir(raw_images):
        sys.exit(f"학습할 사진이 없다: {raw_images} (pinky-pull 로 먼저 받을 것)")

    out_dir = make_run_dir()
    print(f"이번 학습 결과 폴더 {out_dir}")

    bootstrap = BEST_PT if os.path.exists(BEST_PT) else YOLOV8S_PT
    print(f"자동 라벨링 모델 {bootstrap}")
    auto_label(raw_dir, bootstrap)

    split_dataset(raw_dir, os.path.join(ROOT, "apple"))
    save_dir = train_model(epochs)

    shutil.copytree(save_dir, out_dir, dirs_exist_ok=True)
    print(f"\n완료 — 이번 학습 스냅샷: {out_dir}")
    print(f"실전에 쓰는 고정 경로도 갱신됨: {BEST_PT}")


if __name__ == "__main__":
    run(*sys.argv[1:])
