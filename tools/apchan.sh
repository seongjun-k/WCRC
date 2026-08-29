#!/usr/bin/env bash
# 로봇 AP 를 지금 이 자리에서 가장 빈 5GHz 채널로 옮긴다. 대회장 도착하면 한 번 돌린다.
#
# 왜 필요한가: 채널 선택은 그 장소의 전파 상황이지 코드가 아니다. 연구실에서 고른 44 가
# 대회장에서도 빈 채널일 이유가 없다. 특히 다른 팀 로봇도 전부 AP 를 띄운다.
#
# 5GHz 는 36/40/44/48 네 개만 쓸 수 있다. 52 이상은 규제상 no-IR(레이더 대역)이라
# hostapd 가 못 올라온다. KR 도메인에서 확인함.
SSH=pinky@192.168.4.1
AP=pinky_1186
USB_MAC=50:3d:d1:bc:0d:76
CANDIDATES="36 40 44 48"

IF=""
for d in /sys/class/net/*; do
  [ "$(cat "$d/address" 2>/dev/null)" = "$USB_MAC" ] && IF=$(basename "$d")
done
[ -n "$IF" ] || { echo "로봇용 USB 랜카드가 안 보인다"; exit 1; }

echo "스캔 중 ($IF)..."
nmcli dev wifi rescan ifname "$IF" >/dev/null 2>&1; sleep 4
SCAN=$(nmcli -t -f SSID,CHAN,SIGNAL dev wifi list ifname "$IF" 2>/dev/null)

echo
echo " 채널   이웃 AP   간섭점수(신호합)"
best=""; best_score=999999
for ch in $CANDIDATES; do
  # 자기 자신은 뺀다. 안 그러면 지금 쓰는 채널이 항상 최악으로 나온다.
  line=$(echo "$SCAN" | awk -F: -v c="$ch" -v me="$AP" '$1!=me && $2==c {n++; s+=$3} END{print n+0, s+0}')
  n=${line% *}; s=${line#* }
  printf "  %-4s  %3d개     %5d\n" "$ch" "$n" "$s"
  [ "$s" -lt "$best_score" ] && { best_score=$s; best=$ch; }
done

cur=$(ssh -o ConnectTimeout=8 "$SSH" "grep '^channel=' /etc/hostapd/hostapd.conf | cut -d= -f2" 2>/dev/null)
echo
echo "지금: ch$cur  ->  추천: ch$best (점수 $best_score)"
[ "$cur" = "$best" ] && { echo "이미 가장 빈 채널이다. 할 일 없음."; exit 0; }

# 5GHz 폭이 80MHz 인 이웃은 36~48 을 통째로 덮는다. nmcli 는 폭을 안 알려주므로
# 점수가 비슷하면 실제 차이가 없을 수 있다. 확실히 나쁠 때만 옮기는 게 낫다.
read -p "ch$cur -> ch$best 로 옮길까? (y/N) " a
[ "$a" = "y" ] || exit 0

echo "로봇 sudo 암호를 물어본다."
ssh -t "$SSH" "sudo sh -c '
  [ -f /etc/hostapd/hostapd.conf.bak24 ] || cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.bak24
  sed -i \"s/^channel=.*/channel=$best/\" /etc/hostapd/hostapd.conf
  rm -f /tmp/ap5_ok
  # 4분 안에 확인이 안 되면 스스로 원복한다. 현장에서 로봇을 잃지 않기 위한 보험.
  setsid sh -c \"sleep 240; [ -f /tmp/ap5_ok ] || { cp /etc/hostapd/hostapd.conf.bak24 /etc/hostapd/hostapd.conf; systemctl restart hostapd; }\" >/dev/null 2>&1 &
  systemctl restart hostapd' " 2>/dev/null

echo -n "재접속 대기"
for i in $(seq 40); do
  sleep 2; printf '.'
  ping -c1 -W2 -I "$IF" 192.168.4.1 >/dev/null 2>&1 && { echo " OK"; break; }
done
if ping -c1 -W2 -I "$IF" 192.168.4.1 >/dev/null 2>&1; then
  ping -c 50 -i 0.05 -W 1 -I "$IF" 192.168.4.1 2>/dev/null | tail -1
  ssh -o ConnectTimeout=8 "$SSH" "touch /tmp/ap5_ok" && echo "ch$best 확정 (자동 원복 취소)"
else
  echo
  echo "못 붙었다. 4분 뒤 로봇이 알아서 원래 채널로 돌아온다. 기다릴 것."
fi
