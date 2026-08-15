"""핑키 로봇 원격 조작 — 로봇의 Jupyter 서버 API를 그대로 쓴다. SSH 자격증명 불필요.

    python tools/robot.py run "ls -al"          # 로봇에서 셸 명령 실행
    python tools/robot.py py  "import cv2"      # 로봇에서 파이썬 코드 실행
    python tools/robot.py put tools/capture.py capture.py
    python tools/robot.py get raw/images/a.jpg raw/images/a.jpg
    python tools/robot.py ls  [경로]
    python tools/robot.py check                 # 연결 왕복 점검

로봇 주소는 PINKY_HOST 로 바꾼다 (기본 192.168.4.1:8888).
"""
import base64
import json
import os
import sys
import uuid

import requests
from websocket import create_connection

HOST = os.environ.get("PINKY_HOST", "192.168.4.1:8888")
API = f"http://{HOST}/api"

S = requests.Session()


def _auth():
    """Jupyter 는 POST/PUT 에 _xsrf 를 요구한다. 페이지를 한 번 열어 쿠키를 받아온다."""
    if "X-XSRFToken" not in S.headers:
        S.get(f"http://{HOST}/tree", timeout=30)
        S.headers["X-XSRFToken"] = S.cookies.get("_xsrf", "")
    return "; ".join(f"{c.name}={c.value}" for c in S.cookies)


def py(code):
    """로봇의 파이썬 커널에서 코드를 실행하고 출력을 돌려준다."""
    cookie = _auth()
    kid = S.post(f"{API}/kernels", json={"name": "python3"}, timeout=30).json()["id"]
    try:
        ws = create_connection(f"ws://{HOST}/api/kernels/{kid}/channels",
                               timeout=60, cookie=cookie)
        msg_id = uuid.uuid4().hex
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": "wcrc", "session": uuid.uuid4().hex,
                       "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "channel": "shell",
        }))
        out = []
        while True:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t, c = m["msg_type"], m["content"]
            if t == "stream":
                out.append(c["text"])
            elif t == "execute_result":
                out.append(c["data"]["text/plain"])
            elif t == "error":
                out.append("\n".join(c["traceback"]))
            elif t == "status" and c["execution_state"] == "idle":
                break
        ws.close()
        return "".join(out)
    finally:
        S.delete(f"{API}/kernels/{kid}", timeout=30)


def run(cmd):
    """로봇에서 셸 명령 실행. 커널을 거치므로 sudo 대화형은 안 된다."""
    return py(
        "import subprocess;r=subprocess.run(%r,shell=True,capture_output=True,text=True);"
        "print(r.stdout+r.stderr,end='')" % cmd
    )


def put(local, remote=None):
    """로컬 파일을 로봇으로 업로드."""
    remote = remote or os.path.basename(local)
    raw = open(local, "rb").read()
    try:
        body = {"type": "file", "format": "text", "content": raw.decode()}
    except UnicodeDecodeError:
        body = {"type": "file", "format": "base64", "content": base64.b64encode(raw).decode()}
    _auth()
    r = S.put(f"{API}/contents/{remote}", json=body, timeout=60)
    r.raise_for_status()
    return f"업로드 {local} -> {HOST}:{remote} ({len(raw)}B)"


def get(remote, local=None):
    """로봇 파일을 내려받는다."""
    local = local or os.path.basename(remote)
    d = S.get(f"{API}/contents/{remote}", params={"content": 1}, timeout=60).json()
    data = base64.b64decode(d["content"]) if d["format"] == "base64" else d["content"].encode()
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    open(local, "wb").write(data)
    return f"다운로드 {HOST}:{remote} -> {local} ({len(data)}B)"


def ls(remote=""):
    d = S.get(f"{API}/contents/{remote}", timeout=30).json()
    return "\n".join(f"{c['type'][:4]:5} {c['name']}" for c in d["content"])


def check():
    """연결 왕복 점검 — 여기서 실패하면 로봇/네트워크 문제다."""
    assert "pong" in run("echo pong"), "셸 실행 실패"
    assert "pong" in py("print('pong')"), "파이썬 실행 실패"
    print(run("echo 계정=$(whoami) 호스트=$(hostname) 홈=$HOME; "
              "python3 -c 'import pinkylib;print(\"pinkylib OK\")' 2>&1 | tail -1"))
    print("check ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    r = {"run": run, "py": py, "put": put, "get": get, "ls": ls, "check": check}[cmd](*sys.argv[2:])
    if r:
        print(r)
