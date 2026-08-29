# WCRC 사과 인식 모델 — 사용법

내 담당은 **`.pt` 모델 하나 만들기**. 주행 코드·Flask 서버는 다른 팀원 담당이다.
왜 이렇게 만드는지(설계 근거)는 [모델학습_정리.md](모델학습_정리.md) 참고.

새 터미널을 열면 alias가 잡혀 있다. 안 잡히면 `source ~/.wcrc_aliases`.

## 0. 먼저 로봇 Wi-Fi에 연결

내 로봇은 **`pinky_1186`** (비번 `wcrc2026pinky`). 로봇이 자기 AP를 쏘고 있고, 이 AP에 붙어야 접속된다.
로봇 주소는 `192.168.4.1` 고정.

```bash
nmcli dev wifi connect pinky_1186 password wcrc2026pinky
probot check             # 접속 확인 (계정=pinky 호스트=raspi 가 나오면 성공)
```

인터넷은 유선(`enp8s0`)으로 계속 나간다. 유선이 없으면 AP에 붙는 동안 인터넷이 끊기니
`wtrain` 같은 다운로드가 필요한 작업은 그 전에 해둔다.

> 근처에 **다른 팀 로봇**(`pinky_d835` 등)이 같이 떠 있다. 붙기 전에 SSID를 꼭 확인할 것.
> 로봇에서 `iw dev | grep ssid` 로 지금 붙은 개체의 AP 이름을 확인할 수 있다.

---

## 전체 흐름

```
로봇에서 촬영 → PC로 가져오기 → 자동 라벨 → 검수 → 분할 → 학습 → 검증 → best.pt 전달
  pinky-cam      pinky-pull      wlabel    labelImg  wsplit  wtrain  wcheck
```

---

## 1. 촬영

로봇을 **사과 세는 지점(마커 앞 정지 위치)** 에 놓고:

```bash
pinky-cam 100 1.0        # 100장, 1초 간격. 로봇의 ~/raw/images 에 쌓인다
```

촬영이 시작되면 **PC 브라우저가 자동으로 열리며 로봇 카메라 화면이 실시간으로 보인다**
(`http://localhost:8080`, 약 20fps). 이걸 보면서 사과가 프레임에 잘 들어오게 로봇을 놓는다.
브라우저가 안 뜨면 그 주소를 직접 열면 된다. 탭을 닫아도 촬영은 계속된다.
(로봇 AP는 22/8888만 열려 있어 8080에 직접 못 붙는다. `pinky-cam`이 SSH 터널로 넘겨준다 —
그래서 `192.168.4.1:8080`이 아니라 **localhost**:8080이다.)

**중간에 멈추려면 `Ctrl+C`.** 터미널이 안 먹거나 백그라운드로 돌고 있으면 다른 터미널에서 `pinky-stop`.

찍는 동안 로봇을 조금씩 옮겨 각도·거리·조명을 바꾼다.
**사과가 없는 배경 프레임도 20장쯤 섞는다** — 오탐(헛박스)을 눌러주는 데 이게 제일 잘 듣는다.

```bash
pinky-pull               # 로봇 ~/raw/images -> 로컬 raw/images
```

## 2. 자동 라벨

```bash
wlabel raw               # raw/images 를 읽어 raw/labels/*.txt 생성
```

기본값은 COCO 사전학습 `yolov8s.pt`의 `apple` 클래스를 쓴다.
**0개만 나오면** 대회 사과가 모형·그림이라 COCO가 못 잡는 것이다 → 아래 "부트스트랩"으로 간다.

## 3. 검수 (생략 금지)

```bash
labelImg raw/images raw/labels
```

- 포맷이 **YOLO**인지 확인 (PascalVOC면 바꾼다)
- 헛박스는 지우고, 놓친 사과는 `w`로 그린다. `Ctrl+S` 저장, `a`/`d` 이동
- **가림 기준을 하나로 정해서 전 이미지에 똑같이 적용** (예: 절반 이상 보이면 라벨)

검수를 건너뛰면 안 되는 이유: 로봇이 3장 찍어 그중 **최댓값**을 채택한다.
한 장의 헛박스가 그대로 최종 점수가 된다.

## 4. 분할 · 학습

```bash
wsplit raw               # 8:2 로 apple/train, apple/val 에 복사
wtrain                   # yolov8s / imgsz=640 / batch=16 / epochs=80
wtrain 120               # epochs 바꾸려면 인자로
```

결과: `runs/detect/train/weights/best.pt`
`CUDA out of memory`가 나면 `tools/dataset.py`의 `batch=16`을 8로 내린다.

## 5. 검증

```bash
wcheck                   # 서버와 같은 conf=0.25 로 "정답 개수 vs 예측 개수" 비교
```

mAP가 아니라 **개수 일치율**이 점수다. 불일치가 나오면 **과다(오탐)부터** 잡는다.
마지막엔 실기 확인: 서버 실행 → `.pt` 드래그앤드롭 → 로봇으로 실제 지점에서 LCD 숫자 확인.

---

## 부트스트랩 (COCO가 사과를 못 잡을 때)

1. 30장만 손으로 라벨 (`labelImg raw/images raw/labels`)
2. `wsplit raw` → `wtrain 50` 으로 초벌 모델을 만든다
3. 그 모델로 나머지를 자동 라벨:
   ```bash
   wlabel raw runs/detect/train/weights/best.pt 0.15
   ```
   (conf를 낮게 줘서 후보를 넉넉히 뽑는다. 지우는 게 그리는 것보다 빠르다)
4. 검수 → `wsplit raw` → `wtrain` 다시

---

## 로봇 조작

```bash
pinky                    # SSH 접속 (pinky@192.168.4.1, 키 등록돼 있어 비번 불필요)
pinky-stop               # 촬영 강제 중지
pinky-jupyter            # 브라우저로 로봇 Jupyter 열기
pinky-push               # capture.py 를 고쳤을 때 로봇에 다시 올리기
probot run "ls -al"      # SSH 대신 Jupyter API로 명령 실행
probot check             # 로봇 연결 점검
```

---

## 문제가 생기면

| 증상 | 원인과 해결 |
|---|---|
| `카메라를 찾을 수 없습니다. 케이블을...` | **케이블 문제가 아니다.** 브라우저를 닫아도 남은 Jupyter 커널이 카메라·모터를 물고 있는 것. `pinky-free` 실행 |
| `현재 카메라가 사용 중입니다` | 같은 원인. `pinky-free` |
| 로봇 접속 안 됨 | AP 연결이 끊긴 것. `nmcli dev wifi connect pinky_1186 password wcrc2026pinky` 후 `probot check` |
| 다른 로봇에 붙은 것 같다 | `pinky 'iw dev \| grep ssid'` 로 AP 이름 확인. `pinky_1186` 이 아니면 잘못 붙은 것 |
| 촬영이 안 멈춘다 | `Ctrl+C`, 안 되면 `pinky-stop` |
| 주행이 안 멈춘다 | `Ctrl+C`, 안 되면 `pinky-kill` |
| 도로를 못 찾는다 | `pinky-run road` 로 추천값 확인 → `drive.py` 의 `ROAD_S_MAX`/`ROAD_V_MIN` 수정 → `pinky-deploy` |
| 프리뷰 화면이 안 열린다 | `http://localhost:8080` 을 직접 연다. 촬영이 시작된 뒤에만 열린다 |
| `wlabel` 결과가 전부 0개 | COCO가 이 사과를 모른다 → 부트스트랩 |
| `CUDA out of memory` | `tools/dataset.py` 의 `batch=16` → 8 |
| alias가 없다고 나옴 | 새 터미널을 열거나 `source ~/.wcrc_aliases` |
| 개수가 실제보다 많게 나옴 | 오탐. 배경만 찍은 네거티브 이미지를 늘려 재학습 |

---

## 주행 코드

`drive.py` — 배포 템플릿을 실주행 가능하게 고쳐 **단일 스크립트**로 만든 것.
주피터를 안 쓴다: 브라우저를 닫아도 커널이 남아 카메라·모터를 물고 있어서
"카메라를 찾을 수 없습니다" 가 뜬다. 스크립트는 끝나면서 하드웨어를 반납한다.

```bash
pinky-deploy                 # PC -> 로봇으로 drive.py 복사
pinky-run check              # 출발 전 점검 (모터 안 돎)
pinky-run run                # ★ 실제 주행
pinky-run pose               # 마커 x, z 읽기 -> target_list 채우기
pinky-run forward-cal        # MOVE_FORWARD_PER_ONE 측정
pinky-run turn-cal           # TURN_QUATER_TIME / TURN_HALF_TIME 측정
pinky-run road               # 도로 마스크가 지금 화면에서 되는지
```

```bash
python tools/sim_drive.py     # 하드웨어 없이 주행 로직 검증 (7가지 시나리오)
```

```bash
python tools/road.py          # 도로 마스크 + 전진 분할 검증
```

로봇을 코스에 올리기 전에 먼저 돌린다. 대회 기회가 4번뿐이라 코스에서 처음 확인하는
건 낭비다. 둘 다 pinkylib 을 가짜로 물리고 `drive.py` 를 그대로 실행한다 —
**drive.py 를 고치면 같이 검증된다.**

검증 항목: 정상 주행 / 회전 ±50도 어긋남 / 방해 마커 / 마커 전무 / 간헐 인식 실패 /
주행 중 마커 소실 / 정렬 중 영구 소실. 전부 "무한루프 없이 끝나는가"를 본다
(규정: 60초 정체 시 기회 종료).

### 코스에서 실측해야 하는 값 (지금은 전부 임시값)

| 무엇 | 방법 |
|---|---|
| `my_ip` | PC 에서 `ip a` / `ipconfig` |
| `MOVE_FORWARD_PER_ONE` | `pinky-run forward-cal` |
| `TURN_HALF_TIME` / `TURN_QUATER_TIME` | `pinky-run turn-cal` |
| `target_list` 의 x, z | `pinky-run pose` — 마커별로 로봇 세우고 읽기 |
| `after_track_list` 회전 방향 | 코스 보며 조정 (제일 흔한 실수) |
| `ROAD_S_MAX` / `ROAD_V_MIN` | `pinky-run road` — 도로 위에서 |
| `TO_CROSSWALK_TIME` | 하차장 → 횡단보도 전진 시간 |

고쳤으면 `pinky-deploy` 로 다시 올리고, 출발 전 `pinky-run check` 로 전부 OK 확인.

> `set_calibration()` 은 상대경로가 기본값이라 로봇 홈에서는 파일을 못 찾는다.
> `CALIBRATION_PATH = /home/pinky/CH/camera_calibration.npz` 로 박아뒀다.

---

## 파일 구조

```
apple/                  데이터셋 (data.yaml, train/, val/)
raw/                    촬영 원본 + 자동 라벨 (images/, labels/)
tools/
  capture.py            로봇에서 실행하는 촬영 스크립트
  dataset.py            label / split / train / check / demo
  robot.py              로봇 원격 조작 (Jupyter API)
  sim_drive.py          주행 로직 시뮬레이터 (하드웨어 없이 drive.py 검증)
  road.py               도로 마스크 튜닝·검증

drive.py                주행 코드 (로봇에서 실행하는 단일 스크립트)
전략.md                  주행 전략과 당일 순서
모델학습_정리.md         설계 근거와 준비자료 분석
runs/detect/train/weights/best.pt    학습 결과물
```

alias 정의는 `~/.wcrc_aliases`에 있다. 로봇 주소가 바뀌면 그 파일만 고치면 된다.
