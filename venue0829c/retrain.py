"""라벨링 끝난 venue0829c 를 학습셋에 넣고 현재 best.pt 에서 이어 학습한다.
   실행:  .venv/bin/python venue0829c/retrain.py
"""
import glob, os, shutil, sys, time
ROOT = "/home/ksj/orca/projects/WCRC"
SRC  = os.path.join(ROOT, "venue0829c")
BEST = os.path.join(ROOT, "runs/detect/train/weights/best.pt")
OVERSAMPLE = 3          # 어려운 케이스라 3배로 넣는다 (_r1,_r2 사본)
VAL_EVERY  = 6          # 6장에 1장 val

def add():
    imgs = sorted(glob.glob(os.path.join(SRC, "images", "*.jpg")))
    if not imgs: sys.exit("images 가 비었다")
    n = {"train":0, "val":0}
    for i, img in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl  = os.path.join(SRC, "labels", stem + ".txt")
        txt  = open(lbl).read() if os.path.exists(lbl) else ""   # 없으면 사과 0개(네거티브)
        part = "val" if i % VAL_EVERY == 0 else "train"
        # 원본 1장
        shutil.copy2(img, f"{ROOT}/apple/{part}/images/{stem}.jpg")
        open(f"{ROOT}/apple/{part}/labels/{stem}.txt","w").write(txt)
        n[part]+=1
        # 오버샘플 사본은 train 에만 (val 중복 금지)
        if part=="train":
            for r in range(1, OVERSAMPLE):
                shutil.copy2(img, f"{ROOT}/apple/train/images/{stem}_r{r}.jpg")
                open(f"{ROOT}/apple/train/labels/{stem}_r{r}.txt","w").write(txt)
                n["train"]+=1
    print(f"추가: train {n['train']}장 / val {n['val']}장")

def train():
    from ultralytics import YOLO
    bak = BEST.replace("best.pt", f"best_before_1cha_{time.strftime('%H%M')}.pt")
    if os.path.exists(BEST):
        shutil.copy2(BEST, bak); print(f"현 모델 백업 -> {bak}")
    base = BEST if os.path.exists(BEST) else os.path.join(ROOT,"pt_file/yolov8s.pt")
    print(f"이어 학습 base = {base}")
    # 이어 학습이라 짧게. 급하면 이대로(약 5분), 화질 더 원하면 imgsz=960 epochs=60.
    ep  = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    isz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    r = YOLO(base).train(data=os.path.join(ROOT,"apple/data.yaml"),
                         epochs=ep, imgsz=isz, batch=16, device=0,
                         patience=0, lr0=0.001, freeze=10)
    src = os.path.join(str(r.save_dir), "weights", "best.pt")
    shutil.copy2(src, BEST)
    print(f"\n{src}\n-> {BEST}  (서버 재시작하면 적용)")

if __name__ == "__main__":
    add(); train()
