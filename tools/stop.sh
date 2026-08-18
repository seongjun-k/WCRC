#!/usr/bin/env bash
# 주행 강제 중지. [d]rive.py 로 쓰는 이유: 이 SSH 명령줄 자신이 매칭돼 자기를 죽이는 걸 피한다.
ssh -o ConnectTimeout=5 pinky@192.168.4.1 \
  'for p in $(pgrep -f "[d]rive.py"); do kill $p && echo "주행 중지 (pid $p)"; done
   pgrep -f "[d]rive.py" >/dev/null || echo "실행 중인 주행 없음"' \
  || echo "로봇에 연결할 수 없다"
sleep 2
