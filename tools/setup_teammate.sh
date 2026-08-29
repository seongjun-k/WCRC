#!/usr/bin/env bash
# 팀원 노트북에 대회 실행 환경을 통째로 깐다. 내 노트북에서 돌린다.
#
#     bash tools/setup_teammate.sh lsk0522@100.95.167.96
#
# 하는 일: 코드+모델 복사 -> venv 와 파이썬 패키지 -> alias -> 바탕화면 아이콘.
# 인터넷이 필요하다 (대회장 가기 전에 끝낼 것).
# 로봇 ssh 키는 로봇 AP 에 붙어야 되므로 팀원 노트북에서 따로 한다 (맨 끝에 안내).
set -u
DEST=${1:?"쓰는 법: bash tools/setup_teammate.sh 사용자@주소"}
SRC=$(cd "$(dirname "$0")/.." && pwd)
REMOTE_DIR=WCRC

red() { printf '\033[31m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }

ssh -o ConnectTimeout=8 "$DEST" true 2>/dev/null \
  || { red "ssh 로 못 붙는다. 팀원 노트북에서 먼저:"
       echo "  sudo apt install -y openssh-server && sudo systemctl enable --now ssh"; exit 1; }

# 1. 코드와 모델. 학습 데이터(raw, apple)와 지난 추론 결과는 안 보낸다 — 대회에 안 쓴다.
echo "[1/4] 코드·모델 복사"
# runs 통째로는 262MB 라 뺀다. 필요한 건 best.pt 하나뿐이다.
rsync -a --delete --info=stats1 \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'raw' --exclude 'apple' --exclude 'labelcheck' --exclude 'runs' \
  "$SRC/" "$DEST:$REMOTE_DIR/" || { red "복사 실패"; exit 1; }
ssh "$DEST" "mkdir -p $REMOTE_DIR/runs/detect/train/weights"
rsync -a "$SRC/runs/detect/train/weights/best.pt" \
  "$DEST:$REMOTE_DIR/runs/detect/train/weights/best.pt" || { red "모델 복사 실패"; exit 1; }

# 2. venv. NVIDIA 가 있으면 CUDA 판, 없으면 CPU 판(200MB)을 깐다.
#    CPU 로도 돌긴 하는데 확대 추론이 장당 1초를 넘으면 APPLE_JOIN_TIMEOUT 6초에 걸린다.
echo "[2/4] 파이썬 환경 (몇 분 걸린다)"
ssh "$DEST" "bash -s" <<'REMOTE' || { red "설치 실패"; exit 1; }
set -e
cd ~/WCRC
command -v python3 >/dev/null || { echo "python3 가 없다"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip -q install --upgrade pip
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  echo "  NVIDIA GPU 감지 — CUDA 판 torch"
  .venv/bin/pip -q install torch torchvision
else
  echo "  GPU 없음 — CPU 판 torch"
  .venv/bin/pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/pip -q install ultralytics flask opencv-python requests pillow numpy
.venv/bin/python -c "
import os; os.environ['YOLO_OFFLINE']='1'
from ultralytics import YOLO; m=YOLO('runs/detect/train/weights/best.pt')
print('모델 OK', m.names)"
REMOTE

# 3. alias + 바탕화면 아이콘. 경로는 팀원 홈 기준으로 다시 쓴다.
echo "[3/4] alias · 바탕화면 아이콘"
ssh "$DEST" "bash -s" <<'REMOTE'
set -e
W=$HOME/WCRC
cat > ~/.wcrc_aliases <<A
export WCRC=$W
export PINKY_SSH=pinky@192.168.4.1
alias wrun='bash \$WCRC/tools/go.sh'
alias wruna='bash \$WCRC/tools/go.sh auto'
alias wstop='bash \$WCRC/tools/stop.sh'
wtune() { ( cd \$WCRC && .venv/bin/python tools/tune.py "\$@" ); }
wchan() { ( cd \$WCRC && bash tools/apchan.sh ); }
wautotrain() { ( cd \$WCRC && .venv/bin/python tools/auto_train.py "\$@" ); }
A
grep -q wcrc_aliases ~/.bashrc || echo '[ -f ~/.wcrc_aliases ] && source ~/.wcrc_aliases' >> ~/.bashrc

# 바탕화면 폴더 이름이 한글일 수도 영어일 수도 있다
D=$(xdg-user-dir DESKTOP 2>/dev/null); [ -d "$D" ] || D=$HOME/Desktop; mkdir -p "$D"
cat > "$D/WCRC 주행.desktop" <<A
[Desktop Entry]
Type=Application
Name=WCRC 주행
Comment=로봇 연결 · 서버 · 코드 업로드 · 점검 · 주행
Exec=env WCRC_LAUNCHER=1 gnome-terminal --title="WCRC 주행" --geometry=100x35 -- bash $W/tools/go.sh
Icon=applications-engineering
Terminal=false
Categories=Utility;
A
cat > "$D/WCRC 정지.desktop" <<A
[Desktop Entry]
Type=Application
Name=WCRC 정지
Comment=주행 강제 중지
Exec=gnome-terminal --title="WCRC 정지" --geometry=70x12 -- bash $W/tools/stop.sh
Icon=process-stop
Terminal=false
Categories=Utility;
A
chmod +x "$D"/WCRC*.desktop
gio set "$D/WCRC 주행.desktop" metadata::trusted true 2>/dev/null || true
gio set "$D/WCRC 정지.desktop" metadata::trusted true 2>/dev/null || true
echo "바탕화면: $D"
REMOTE

# 4. 서버가 실제로 뜨는지. 여기까지 되면 로봇만 붙이면 된다.
echo "[4/4] 서버 기동 확인"
ssh "$DEST" "cd ~/WCRC && timeout 90 .venv/bin/python tools/serve.py --selftest" || red "  서버 selftest 실패"

grn "완료. 남은 건 팀원 노트북에서 직접 해야 한다 (로봇이 옆에 있을 때):"
cat <<'TODO'
  1) 로봇 AP 에 붙는다:  nmcli dev wifi connect pinky_1186 password pinkypro
  2) 로봇 ssh 키 (한 번만. 비밀번호는 1):
       ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519   # 이미 있으면 건너뛴다
       ssh-copy-id pinky@192.168.4.1
  3) 바탕화면 "WCRC 주행" 아이콘 우클릭 -> 실행 허용
  4) 점검:  wrun check      (모터 안 돎)
TODO
