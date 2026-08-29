#!/usr/bin/env bash
# 클릭 한 번으로 대회 주행. 바탕화면 런처가 이걸 부른다.
# alias 에 의존하지 않는다 (런처는 .bashrc 를 안 읽는다).
WCRC=$(cd "$(dirname "$0")/.." && pwd)   # 스크립트 위치 기준. 노트북마다 경로가 다르다
SSH=pinky@192.168.4.1
AP=pinky_1186
# 로봇 전용 USB 랜카드. 노트북 내장 와이파이는 인터넷에 그대로 둔다.
# MAC 으로 찾는 이유: 인터페이스 이름(wlx...)이 바뀌어도 따라간다.
# USB 가 없는 노트북(팀원 것)은 pick_if 가 내장 와이파이로 떨어진다.
USB_MAC=${WCRC_ROBOT_MAC:-50:3d:d1:bc:0d:76}

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

wifi_ifs() { nmcli -t -f DEVICE,TYPE dev 2>/dev/null | awk -F: '$2=="wifi"{print $1}'; }

# 로봇에 붙일 카드: USB -> 이미 192.168.4.x 를 물고 있는 카드 -> 첫 번째 와이파이
pick_if() {
  usb_if && return 0
  local d
  for d in $(wifi_ifs); do
    ip -o -4 addr show "$d" | grep -q ' 192\.168\.4\.' && { echo "$d"; return 0; }
  done
  wifi_ifs | head -1
}

# 붙었는지는 SSID 이름이 아니라 "로봇이 응답하나" 로 판단한다.
# nmcli 가 돌려주는 건 프로필 이름이라 재접속 때마다 "pinky_1186 2" 처럼 늘어나
# 이름 비교가 어긋난다 (실제로 겪었다). ping 은 그런 게 없다.
robot_up() { ping -c1 -W2 -I "$1" 192.168.4.1 >/dev/null 2>&1; }

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
# 바탕화면 런처로 열렸을 때만 끝에서 멈춘다. 안 그러면 창이 그냥 사라져서
# 왜 실패했는지 못 본다. 터미널에서 `wrun` 으로 부른 경우엔 그냥 끝낸다.
hold() { [ -n "$WCRC_LAUNCHER" ] && { echo; echo "10초 후 창이 자동으로 닫힙니다 (엔터 안 눌러도 됨)"; read -t 10 -p "" _; }; }
die()  { red "$*"; hold; exit 1; }

echo "════════════════════════════════════"
echo "  WCRC 주행$([ "$1" = check ] && echo ' — 점검만 (모터 안 돎)')$([ "$1" = auto ] && echo ' — 자동 (엔터 없음)')"
echo "════════════════════════════════════"

# 1. 로봇 AP — USB 랜카드로만 붙인다
IF=$(pick_if)
[ -n "$IF" ] || die "와이파이 카드를 못 찾았다. USB 랜카드를 꽂거나 와이파이를 켤 것."

# ★ USB 말고 다른 카드가 192.168.4.x 를 물고 있으면 경로가 둘이 된다. 그러면 로봇이
#   보낸 사진의 응답이 엉뚱한 카드로 나가서 업로드가 10초 넘게 멈춘다 (실제로 겪었다 —
#   사과 사진 6장이 12초 밀렸다). GUI 네트워크 메뉴에서 pinky_1186 을 클릭하면 이렇게 된다.
#   AP 이름이 아니라 "주소"로 판단하는 이유: 대회장의 남의 AP 가 같은 대역을 쓸 수도 있다.
for d in $(ip -o -4 addr show | awk '$4 ~ /^192\.168\.4\./ {print $2}'); do
  [ "$d" = "$IF" ] && continue
  red "  $d 가 192.168.4.x 를 물고 있다 — 끊는다 (로봇과 경로가 겹친다)"
  nmcli dev disconnect "$d" >/dev/null 2>&1
done

if ! robot_up "$IF"; then
  echo "[1/4] 로봇 와이파이에 붙는 중 ($IF)..."
  # 저장된 프로필을 올린다. `dev wifi connect` 를 쓰면 실패할 때마다 프로필이
  # 하나씩 새로 생기고, 새 프로필은 USB 고정이 안 걸려 있어 내장 카드를 물 수 있다.
  nmcli con up "$AP" ifname "$IF" >/dev/null 2>&1 \
    || nmcli dev wifi connect "$AP" password wcrc2026pinky ifname "$IF" >/dev/null 2>&1
  for _ in 1 2 3 4 5 6; do robot_up "$IF" && break; sleep 1; done
  robot_up "$IF" || die "로봇($AP / 192.168.4.1)에 못 붙었다. 로봇 전원과 USB 를 확인할 것."
fi
grn "[1/4] 로봇 연결 OK ($IF)"

# 2. 서버
# tune.py 가 전처리 값을 새로 골랐는데 서버가 옛날 값으로 떠 있으면 아무 소용이 없다.
# serve.py 는 tune.json 을 기동할 때 한 번만 읽으므로, 더 새로우면 다시 띄운다.
SRV_PID=$(ss -ltnp 2>/dev/null | grep ':5000' | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$SRV_PID" ] && [ -f "$WCRC/tools/tune.json" ]; then
  started=$(( $(date +%s) - $(ps -o etimes= -p "$SRV_PID" | tr -d ' ') ))
  if [ "$(stat -c %Y "$WCRC/tools/tune.json")" -gt "$started" ]; then
    echo "[2/4] 튜닝 값이 서버보다 새롭다 — 서버 다시 띄운다"
    kill "$SRV_PID"; sleep 1
  fi
fi

if ! curl -s -m 3 -o /dev/null http://127.0.0.1:5000/; then
  echo "[2/4] 사과 인식 서버 기동 중..."
  (cd "$WCRC" && nohup .venv/bin/python tools/serve.py >/tmp/wcrc-serve.log 2>&1 &)
  # 20초로 끊었다가 실패했다. 모델 로드는 2초면 끝나지만 노트북이 다른 이유로
  # 느릴 수 있으니 여유를 준다. 어차피 뜨는 즉시 빠져나온다.
  for i in $(seq 60); do
    sleep 1
    curl -s -m 2 -o /dev/null http://127.0.0.1:5000/ && break
    printf '.'
  done
  echo
fi
curl -s -m 3 -o /dev/null http://127.0.0.1:5000/ \
  || die "서버 기동 실패. 로그: tail -30 /tmp/wcrc-serve.log"
grn "[2/4] 서버 OK"

# 이 노트북이 로봇에게 보이는 주소. drive.py 의 my_ip 대신 이걸 쓴다 (노트북이 바뀌어도
# 코드를 안 고친다). 못 구하면 drive.py 기본값으로 떨어진다.
MY_IP=$(ip -o -4 addr show "$IF" | awk '{print $4}' | cut -d/ -f1 | head -1)
ENVS=${MY_IP:+WCRC_SERVER_IP=$MY_IP}
[ -n "$MY_IP" ] && echo "  서버 주소 $MY_IP:5000 (로봇에 전달)"

# 3. 코드 업로드 (겸 시계 맞추기 — 로봇에 RTC 가 없어 부팅할 때마다 12일씩 틀어진다.
#    사진 폴더 이름이 날짜라서 틀리면 어느 주행 건지 못 찾는다. 대회장엔 NTP 도 없다)
ssh -o ConnectTimeout=8 "$SSH" "echo 1 | sudo -S date -s '$(date '+%Y-%m-%d %H:%M:%S')'" \
  >/dev/null 2>&1 || red "  시계 동기 실패 (주행에는 지장 없음)"
scp -q -o ConnectTimeout=8 "$WCRC/drive.py" "$SSH:~/drive.py" || die "코드 업로드 실패"
grn "[3/4] 코드 업로드 OK"

# 4. 출발 전 점검
echo "[4/4] 출발 전 점검..."
echo "────────────────────────────────────"
ssh -o ConnectTimeout=8 "$SSH" "cd ~ && $ENVS python3 drive.py check" 2>&1 | grep -v '^\['
echo "────────────────────────────────────"

# `wrun check` 는 점검까지만 하고 끝낸다 (모터가 안 돈다)
if [ "$1" = check ]; then hold; exit 0; fi

# 5번에서 "방금 찍은 폴더"를 이름(=로봇 시계)순이 아니라 "달리기 전엔 없던 폴더"로
# 찾기 위한 스냅샷. 여기서 찍어둬야 이 뒤에 drive.py 가 만드는 폴더와 구분된다.
BEFORE=$(ssh -o ConnectTimeout=8 "$SSH" "ls -d ~/apple_shots/*/ 2>/dev/null")

# `wrun auto` (= wruna) 는 drive.py 의 엔터 대기(wait_for_enter)까지 "now" 인자로
# 건너뛴다 — 사람이 옆에서 엔터를 눌러줄 필요 없이 끝까지 무인으로 돈다.
if [ "$1" = auto ]; then
  echo
  echo "위에 FAIL 이 없으면 로봇을 START 선 안에 놓을 것."
  echo "자동 모드 — 엔터 없이 곧바로 출발한다."
  echo
  ssh -t "$SSH" "cd ~ && $ENVS python3 drive.py run now"
else
  # 여기서 엔터를 한 번 더 받지 않는다. 엔터가 둘이면 종 울리고 누른 첫 엔터가
  # ssh 접속 + 하드웨어 초기화(수 초)로 날아간다. drive.py 가 준비를 다 끝내고
  # 엔터 하나에서 곧바로 출발하도록 되어 있으니 그걸 그대로 쓴다.
  echo
  echo "위에 FAIL 이 없으면 로봇을 START 선 안에 놓을 것."
  echo "아래에서 준비가 끝나면 멈춘다. 심판이 출발 신호를 주면 그때 엔터 = 즉시 출발."
  echo
  ssh -t "$SSH" "cd ~ && $ENVS python3 drive.py run"
fi

# 5. 방금 찍은 사과 사진을 PC 로. $WCRC/apple_shots/<날짜시간횟수>/ 에 원본 파일명
# 그대로 저장한다 (로봇 쪽 ~/apple_shots 구조를 그대로 미러링). 실패해도 주행 자체는
# 이미 끝난 뒤라 여기서 죽이지 않는다 — 경고만 하고 넘어간다.
#
# ★ 폴더를 "이름(날짜)순 마지막"으로 고르지 않는다. 로봇에 RTC 가 없어 시계 동기(3번)가
#   가끔 실패하면 방금 찍은 폴더 이름이 기존 폴더보다 "작게" 나와서 이름순 마지막이
#   옛날 폴더를 가리키고, 방금 사진은 로봇에만 남고 영영 못 받아온다 (실제로 겪었다 —
#   시계가 안 맞았던 회차 9개가 로봇에만 남아 있었다). 그래서 BEFORE 와 비교해서
#   "이번에 새로 생긴 폴더"로 고른다 — 로봇 시계가 틀려도 항상 맞는다.
echo "방금 찍은 사진 받는 중..."
AFTER=$(ssh -o ConnectTimeout=8 "$SSH" "ls -d ~/apple_shots/*/ 2>/dev/null")
LATEST=$(comm -13 <(printf '%s\n' "$BEFORE" | sort) <(printf '%s\n' "$AFTER" | sort) | tail -1)
[ -z "$LATEST" ] && LATEST=$(printf '%s\n' "$AFTER" | tail -1)   # 새 폴더를 못 가르면 예전 방식으로 폴백
if [ -z "$LATEST" ]; then
  red "  사진 폴더를 못 찾았다 — 나중에 pinky-pull 로 직접 받을 것"
else
  RUNDIR=$(basename "$LATEST")
  mkdir -p "$WCRC/apple_shots"
  rm -rf "$WCRC/apple_shots/$RUNDIR"   # 재시도 시 부분 복사가 안 섞이게
  if scp -r -q "$SSH:$LATEST" "$WCRC/apple_shots/" \
      && [ -n "$(ls -A "$WCRC/apple_shots/$RUNDIR" 2>/dev/null)" ]; then
    N=$(ls "$WCRC/apple_shots/$RUNDIR" | wc -l)
    grn "  ${N}장 -> apple_shots/${RUNDIR}/"

    # 인식이 뭘 보고 셌는지 바로 눈으로 확인할 수 있게, 박스 그려진 사본도 같이 남긴다.
    # 서버가 그때 쓰던 모델 그대로 다시 태워서 그리는 것이라 실제 채점과 같은 결과다.
    PREDICT_DIR=$(grep -oP '^결과\s+\K.*' /tmp/wcrc-serve.log 2>/dev/null | tail -1)
    if [ -n "$PREDICT_DIR" ] && curl -s -m 3 -o /dev/null http://127.0.0.1:5000/; then
      mkdir -p "$WCRC/apple_shots/$RUNDIR/boxed"
      for f in "$WCRC/apple_shots/$RUNDIR"/*.jpg; do
        saved=$(curl -s -X POST -F "image=@$f" http://127.0.0.1:5000/predict \
                | python3 -c "import json,sys; print(json.load(sys.stdin).get('saved_filename',''))" 2>/dev/null)
        [ -n "$saved" ] && [ -f "$PREDICT_DIR/$saved" ] \
          && cp "$PREDICT_DIR/$saved" "$WCRC/apple_shots/$RUNDIR/boxed/$(basename "$f")"
      done
      grn "  박스 그려진 사본 -> apple_shots/${RUNDIR}/boxed/"
    else
      red "  서버가 안 떠 있어 박스 사본은 못 만들었다"
    fi
  else
    red "  사진을 못 받았다 — 나중에 pinky-pull 로 직접 받을 것"
  fi
fi

echo
hold
