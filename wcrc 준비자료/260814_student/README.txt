=====================================================
 2026 WCRC 학생용 템플릿 코드 (Pinky pro)
=====================================================

[폴더 구성]
  flask_server_wcrc/
    app.py    : YOLO 추론 Flask 서버 (PC에서 실행)
    run.bat   : 서버 실행용 배치 파일 (더블 클릭)
  pinky_pro_wcrc/
    pinkypro_wcrc_student.ipynb : 로봇용 학생 템플릿 (Jupyter, 권장)
    pinkypro_wcrc_student.py    : 위와 동일한 코드의 스크립트 버전

[실행 순서]
  1. PC에서 run.bat 더블 클릭 → Flask 서버 실행 (브라우저 자동 실행)
  2. 브라우저에 학습시킨 .pt 모델 파일을 드래그 앤 드롭
  3. PC의 cmd 창에서 ipconfig 로 IP 주소 확인
  4. 로봇에 접속해 Jupyter Notebook 에서 pinkypro_wcrc_student.ipynb 열기
  5. my_ip 에 PC IP 입력 → 셀을 위에서부터 순서대로 실행

[학생이 수정하는 곳은 딱 4곳!]
  코드에서 "학생 수정" 으로 검색(Ctrl+F)하면 바로 찾을 수 있습니다.
  [학생 수정 (1)] my_ip                : Flask 서버 PC의 IP
  [학생 수정 (2)] MOVE_FORWARD_PER_ONE : 본인 로봇으로 측정 (노트북 맨 아래 부록 셀)
  [학생 수정 (3)] target_list          : 주행 순서대로 마커 id / x / z / 탐색 방향
  [학생 수정 (4)] after_track_list     : 각 마커 도착 후 동작 (target_list 와 같은 개수!)
  이 4곳 외의 코드는 수정하지 않습니다. (읽고 이해하는 것이 목표)

자세한 내용은 함께 배포된 "WCRC 학생 실습 가이드" 문서를 참고하세요.
