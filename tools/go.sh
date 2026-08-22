#!/usr/bin/env bash
# 클릭 한 번으로 대회 주행. 바탕화면 런처가 이걸 부른다.
# alias 에 의존하지 않는다 (런처는 .bashrc 를 안 읽는다).
WCRC=/home/ksj/orca/projects/WCRC
SSH=pinky@192.168.4.1
AP=pinky_1186
# 로봇 전용 USB 랜카드. 노트북 내장 와이파이는 인터넷에 그대로 둔다.
# MAC 으로 찾는 이유: 인터페이스 이름(wlx...)이 바뀌어도 따라간다.
USB_MAC=50:3d:d1:bc:0d:76

# ★ iwgetid -r 을 쓰면 안 된다. 랜카드가 둘이면 첫 번째(내장) SSID 만 돌려줘서
#   USB 가 이미 로봇에 붙어 있어도 "안 붙었다" 로 읽고, ifname 없이 재접속을 걸어
#   내장 카드를 로봇 AP 로 끌고 간다 = 인터넷이 끊긴다.
usb_if() {
  local d
  for d in /sys/class/net/*; do
    [ "$(cat "$d/address" 2>/dev/null)" = "$USB_MAC" ] && { basename "$d"; return 0; }
  done
  return 1
}

# 붙었는지는 SSID 이름이 아니라 "로봇이 응답하나" 로 판단한다.
# nmcli 가 돌려주는 건 프로필 이름이라 재접속 때마다 "pinky_1186 2" 처럼 늘어나
# 이름 비교가 어긋난다 (실제로 겪었다). ping 은 그런 게 없다.
robot_up() { ping -c1 -W2 -I "$1" 192.168.4.1 >/dev/null 2>&1; }

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
# 바탕화면 런처로 열렸을 때만 끝에서 멈춘다. 안 그러면 창이 그냥 사라져서
# 왜 실패했는지 못 본다. 터미널에서 `wrun` 으로 부른 경우엔 그냥 끝낸다.
hold() { [ -n "$WCRC_LAUNCHER" ] && { echo; read -p "엔터를 누르면 창을 닫습니다 " _; }; }
die()  { red "$*"; hold; exit 1; }

echo "════════════════════════════════════"
echo "  WCRC 주행$([ "$1" = check ] && echo ' — 점검만 (모터 안 돎)')"
echo "════════════════════════════════════"

# 1. 로봇 AP — USB 랜카드로만 붙인다
IF=$(usb_if) || die "로봇용 USB 랜카드가 안 보인다. USB 를 꽂고 다시 실행할 것."
if ! robot_up "$IF"; then
  echo "[1/4] 로봇 와이파이에 붙는 중 ($IF)..."
  # 저장된 프로필을 올린다. `dev wifi connect` 를 쓰면 실패할 때마다 프로필이
  # 하나씩 새로 생기고, 새 프로필은 USB 고정이 안 걸려 있어 내장 카드를 물 수 있다.
  nmcli con up "$AP" ifname "$IF" >/dev/null 2>&1 \
    || nmcli dev wifi connect "$AP" password pinkypro ifname "$IF" >/dev/null 2>&1
  for _ in 1 2 3 4 5 6; do robot_up "$IF" && break; sleep 1; done
  robot_up "$IF" || die "로봇($AP / 192.168.4.1)에 못 붙었다. 로봇 전원과 USB 를 확인할 것."
fi
grn "[1/4] 로봇 연결 OK ($IF)"

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
