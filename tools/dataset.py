"""데이터셋 자동 라벨링 / 분할 / 학습. PC(RTX 4060)에서 실행.

    python tools/dataset.py label raw [모델] [conf]   # raw/images -> raw/labels (YOLO txt)
    python tools/dataset.py split raw                 # raw -> apple/train, apple/val (8:2)
    python tools/dataset.py train [epochs]            # apple/data.yaml 로 학습
    python tools/dataset.py check [모델]              # 개수 일치율 검증 (val 기준)
    python tools/dataset.py demo                      # 자체 점검

label 은 라벨 "초안"만 만든다. labelImg 로 열어 반드시 검수할 것.
서버는 박스 개수만 세므로(app.py) 헛박스 하나가 그대로 점수가 된다.
"""
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "apple", "data.yaml")


def label(src="raw", model_path="yolov8s.pt", conf=0.25):
    conf = float(conf)
    """모델로 예측해 YOLO 포맷 라벨 초안 생성.

    기본 yolov8s.pt 는 COCO 사전학습 모델이라 'apple' 클래스가 이미 있다.
    잘 안 잡히면(모형/그림 사과) 30장쯤 손으로 라벨해 학습한 뒤
    그 best.pt 를 모델로 넘겨 나머지를 자동 라벨한다(부트스트랩).

    이미 라벨이 있는(파일 크기 > 0) 이미지는 건너뛴다.
    """
    from ultralytics import YOLO

    img_dir, lbl_dir = os.path.join(src, "images"), os.path.join(src, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    # 기존 라벨 백업
    bak_dir = lbl_dir + "_bak"
    if os.path.isdir(lbl_dir) and any(f.endswith(".txt") for f in os.listdir(lbl_dir)):
        if os.path.isdir(bak_dir):
            shutil.rmtree(bak_dir)
        shutil.copytree(lbl_dir, bak_dir)
        print(f"기존 라벨 백업 -> {bak_dir}")

    model = YOLO(model_path)
    # COCO 모델이면 apple 클래스만 남긴다. 커스텀 1클래스 모델이면 전부 사용.
    keep = {i for i, n in model.names.items() if n == "apple"} or set(model.names)
    print(f"model={model_path} keep={[model.names[i] for i in sorted(keep)]} conf={conf}")

    total, skipped = 0, 0
    for r in model.predict(source=img_dir, conf=conf, stream=True, verbose=False):
        stem = os.path.splitext(os.path.basename(r.path))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        # 이미 수동 라벨이 있으면 건너뛴다
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            skipped += 1
            print(f"{stem}: 기존 라벨 유지 (건너뜀)")
            continue
        # xywhn = 이미지 크기로 정규화된 (중심x, 중심y, 너비, 높이) — YOLO txt 형식 그대로
        lines = [
            "0 " + " ".join(f"{v:.6f}" for v in box)
            for box, c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist())
            if int(c) in keep
        ]
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))
        total += len(lines)
        print(f"{stem}: {len(lines)}개")

    print(f"\n박스 {total}개 생성 (건너뜀 {skipped}장) -> {lbl_dir}\nlabelImg 로 열어 검수한 뒤 split 실행")


def split(src="raw", dst=None, val_every=5):
    """8:2 분할. 정렬 후 5장마다 1장을 val 로 — 시드 없이 재현 가능."""
    dst = dst or os.path.join(ROOT, "apple")
    imgs = sorted(glob.glob(os.path.join(src, "images", "*.jpg")))
    if not imgs:
        sys.exit(f"이미지가 없다: {src}/images")

    n = {"train": 0, "val": 0}
    for i, img in enumerate(imgs):
        part = "val" if i % val_every == 0 else "train"
        stem = os.path.splitext(os.path.basename(img))[0]
        shutil.copy2(img, os.path.join(dst, part, "images", os.path.basename(img)))
        lbl = os.path.join(src, "labels", stem + ".txt")
        # 라벨 없는 이미지 = 사과 0개(네거티브). 빈 txt 를 만들어야 학습에 반영된다
        open(os.path.join(dst, part, "labels", stem + ".txt"), "w").write(
            open(lbl).read() if os.path.exists(lbl) else ""
        )
        n[part] += 1

    print(f"train {n['train']}장 / val {n['val']}장 -> {dst}")


def train(epochs=80):
    """RTX 4060 Laptop(8GB) 기준 설정."""
    from ultralytics import YOLO

    r = YOLO("yolov8s.pt").train(
        data=DATA_YAML,
        epochs=int(epochs),
        imgsz=640,   # 서버가 imgsz 지정 없이 추론하므로 기본값 640 에 맞춘다
        batch=16,    # 8GB 기준. CUDA out of memory 나면 8 로 내린다
        device=0,
        patience=0,       # EarlyStopping 끄기 — 끝까지 학습
        lr0=0.001,        # 소량 데이터 + 소형 객체: 학습률 낮춤
        freeze=10,        # backbone 동결 — COCO 특징 보존, 헤드만 학습
    )

    # ultralytics 는 실행할 때마다 train, train-2, train-3 ... 으로 새 폴더를 만든다.
    # 서버(tools/serve.py)와 wcheck 는 train/weights/best.pt 고정 경로를 보므로
    # 여기서 실제로 복사해 줘야 한다.
    # (예전엔 이 줄이 print 만 하고 복사를 안 해서, 재학습을 해도 계속 옛 모델이
    #  쓰이고 있었다. 성능이 안 바뀌면 제일 먼저 의심할 것.)
    src = os.path.join(str(r.save_dir), "weights", "best.pt")
    dst = os.path.join(ROOT, "runs", "detect", "train", "weights", "best.pt")
    if os.path.abspath(src) != os.path.abspath(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    print(f"\n{src} -> {dst}")


def check(model_path="runs/detect/train/weights/best.pt", conf=0.25):
    """mAP 말고 '개수'로 검증. 서버와 같은 conf=0.25 조건."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    ok = over = under = 0
    for img in sorted(glob.glob(os.path.join(ROOT, "apple", "val", "images", "*.jpg"))):
        lbl = img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        lbl = os.path.splitext(lbl)[0] + ".txt"
        gt = sum(1 for line in open(lbl) if line.strip()) if os.path.exists(lbl) else 0
        pred = len(model(img, conf=conf, verbose=False)[0].boxes)
        if gt == pred:
            ok += 1
        else:
            over += pred > gt
            under += pred < gt
            print(f"{os.path.basename(img)}: 정답 {gt} / 예측 {pred}")
    print(f"\n개수 일치 {ok} / 과다(오탐) {over} / 부족(미탐) {under}")
    print("로봇이 3장 중 최댓값을 쓰므로 과다부터 잡을 것")


def demo():
    """split 로직 자체 점검."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "raw"), os.path.join(tmp, "out")
        os.makedirs(os.path.join(src, "images"))
        os.makedirs(os.path.join(src, "labels"))
        for part in ("train", "val"):
            os.makedirs(os.path.join(dst, part, "images"))
            os.makedirs(os.path.join(dst, part, "labels"))
        for i in range(10):
            open(os.path.join(src, "images", f"{i:02d}.jpg"), "wb").write(b"x")
        open(os.path.join(src, "labels", "00.txt"), "w").write("0 0.5 0.5 0.2 0.2\n")

        split(src, dst)

        assert len(os.listdir(os.path.join(dst, "val", "images"))) == 2
        assert len(os.listdir(os.path.join(dst, "train", "images"))) == 8
        # 라벨 없는 이미지도 빈 txt 가 생겨야 한다 (네거티브 샘플)
        assert len(os.listdir(os.path.join(dst, "train", "labels"))) == 8
        assert open(os.path.join(dst, "val", "labels", "00.txt")).read().strip()
        assert open(os.path.join(dst, "train", "labels", "01.txt")).read() == ""
    print("demo ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    {"label": label, "split": split, "train": train, "check": check, "demo": demo}[cmd](
        *sys.argv[2:]
    )
