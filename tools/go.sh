#!/usr/bin/env bash
# 클릭 한 번으로 대회 주행. 바탕화면 런처가 이걸 부른다.
# alias 에 의존하지 않는다 (런처는 .bashrc 를 안 읽는다).
WCRC=/home/ksj/orca/projects/WCRC
SSH=pinky@192.168.4.1
AP=pinky_1186

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
# 바탕화면 런처로 열렸을 때만 끝에서 멈춘다. 안 그러면 창이 그냥 사라져서
# 왜 실패했는지 못 본다. 터미널에서 `wrun` 으로 부른 경우엔 그냥 끝낸다.
hold() { [ -n "$WCRC_LAUNCHER" ] && { echo; read -p "엔터를 누르면 창을 닫습니다 " _; }; }
die()  { red "$*"; hold; exit 1; }

echo "════════════════════════════════════"
echo "  WCRC 주행$([ "$1" = check ] && echo ' — 점검만 (모터 안 돎)')"
echo "════════════════════════════════════"

# 1. 로봇 AP
if [ "$(iwgetid -r 2>/dev/null)" != "$AP" ]; then
  echo "[1/4] 로봇 와이파이에 붙는 중..."
  nmcli dev wifi connect "$AP" password pinkypro >/dev/null 2>&1
  sleep 3
  [ "$(iwgetid -r 2>/dev/null)" = "$AP" ] || die "로봇 AP($AP)에 못 붙었다. 로봇 전원을 확인할 것."
fi
grn "[1/4] 로봇 연결 OK"

# 2. 서버
if ! curl -s -m 3 -o /dev/null http://127.0.0.1:5000/; then
  echo "[2/4] 사과 인식 서버 기동 중..."
  (cd "$WCRC" && nohup .venv/bin/python tools/serve.py >/tmp/wcrc-serve.log 2>&1 &)
  for i in $(seq 20); do
    sleep 1
    curl -s -m 2 -o /dev/null http://127.0.0.1:5000/ && break
  done
fi
curl -s -m 3 -o /dev/null http://127.0.0.1:5000/ \
  || die "서버 기동 실패. 로그: tail -30 /tmp/wcrc-serve.log"
grn "[2/4] 서버 OK"

# 3. 코드 업로드
scp -q -o ConnectTimeout=8 "$WCRC/drive.py" "$SSH:~/drive.py" || die "코드 업로드 실패"
grn "[3/4] 코드 업로드 OK"

# 4. 출발 전 점검
echo "[4/4] 출발 전 점검..."
echo "────────────────────────────────────"
ssh -o ConnectTimeout=8 "$SSH" "cd ~ && python3 drive.py check" 2>&1 | grep -v '^\['
echo "────────────────────────────────────"

# `wrun check` 는 점검까지만 하고 끝낸다 (모터가 안 돈다)
if [ "$1" = check ]; then hold; exit 0; fi

echo
echo "위에 FAIL 이 없으면 로봇을 START 선 안에 놓고,"
read -p "준비되면 엔터 (그 다음 화면에서 한 번 더 누르면 출발) " _

ssh -t "$SSH" "cd ~ && python3 drive.py run"

echo
hold
