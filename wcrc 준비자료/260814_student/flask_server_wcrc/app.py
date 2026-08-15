import os
import sys
import datetime
import glob
import webbrowser
from threading import Timer
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# 기본 폴더 경로 설정
DIR_PT = r"C:\pinky\pt_file"
DIR_PREDICT = r"C:\pinky\predict_image"

os.makedirs(DIR_PT, exist_ok=True)
os.makedirs(DIR_PREDICT, exist_ok=True)

# 전역 변수
selected_pt_file = None
loaded_model = None

def get_unique_filepath(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    target_path = os.path.join(directory, filename)
    while os.path.exists(target_path):
        target_path = os.path.join(directory, f"{base}({counter}){ext}")
        counter += 1
    return target_path

# --- HTML/CSS/JS (UI 화면) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>YOLOv8 Predict Web Control</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }
        .navbar { background-color: #333; overflow: hidden; display: flex; }
        .dropdown { float: left; overflow: hidden; }
        .dropdown .dropbtn {
            font-size: 16px; border: none; outline: none; color: white;
            padding: 12px 20px; background-color: inherit; cursor: pointer;
        }
        .navbar a:hover, .dropdown:hover .dropbtn { background-color: #555; }
        .dropdown-content {
            display: none; position: absolute; background-color: #f9f9f9;
            min-width: 220px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2); z-index: 100;
        }
        .dropdown-content a {
            float: none; color: black; padding: 12px 16px;
            text-decoration: none; display: block; text-align: left; cursor: pointer;
        }
        .dropdown-content a:hover { background-color: #ddd; }
        .dropdown:hover .dropdown-content { display: block; }

        .container { padding: 20px; max-width: 1100px; margin: 0 auto; }
        .selected-info { font-weight: bold; margin-bottom: 15px; font-size: 15px; color: #2c3e50; }
        
        .drop-zone {
            width: 100%; height: 180px; border: 2px dashed #ccc; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px; color: #777; background-color: #fff;
            transition: all 0.3s ease; box-sizing: border-box; margin-bottom: 25px;
        }
        .drop-zone.dragover { border-color: #333; background-color: #e9e9e9; }
        .drop-zone.error { border-color: #e74c3c; color: #e74c3c; }

        .display-section { display: flex; gap: 20px; height: 400px; }
        
        .file-list-box {
            width: 250px; border: 1px solid #ccc; background: #fff;
            overflow-y: auto; height: 100%; box-sizing: border-box; border-radius: 4px;
        }
        .file-list-box ul { list-style: none; padding: 0; margin: 0; }
        .file-list-box li {
            padding: 10px; font-size: 13px; border-bottom: 1px solid #eee;
            cursor: pointer; word-break: break-all;
        }
        .file-list-box li:hover { background-color: #f1f1f1; }
        .file-list-box li.active { background-color: #007bff; color: white; font-weight: bold; }

        .image-viewer {
            flex: 1; border: 1px solid #ccc; background: #fff;
            display: flex; align-items: center; justify-content: center;
            position: relative; border-radius: 4px; overflow: hidden;
        }
        .image-viewer img { max-width: 100%; max-height: 100%; object-fit: contain; }
        .image-viewer .placeholder-text { color: #aaa; font-size: 20px; font-weight: bold; }
    </style>
</head>
<body>

    <div class="navbar">
        <div class="dropdown">
            <button class="dropbtn">File ▾</button>
            <div class="dropdown-content">
                <a onclick="selectPtFile()">select pt file</a>
                <a onclick="openFolderPath('image')">open image folder path</a>
                <a onclick="openFolderPath('pt')">open pt file path</a>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="selected-info">
            selected file: <span id="selected-pt-path">None</span>
        </div>

        <div class="drop-zone" id="drop-zone">
            Drop your .pt file
        </div>

        <div class="display-section">
            <div class="file-list-box">
                <ul id="image-list"></ul>
            </div>
            <div class="image-viewer" id="image-viewer">
                <span class="placeholder-text" id="viewer-placeholder">Most recent file display</span>
                <img id="viewer-img" src="" style="display:none;">
            </div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const selectedPtPath = document.getElementById('selected-pt-path');
        const imageListEl = document.getElementById('image-list');
        const viewerImg = document.getElementById('viewer-img');
        const viewerPlaceholder = document.getElementById('viewer-placeholder');

        // 상태 관리 변수
        let currentTopFile = "";  // 서버 기준 가장 최신 파일명
        let selectedFile = "";    // 현재 사용자가 선택하여 보고 있는 파일명

        function openFolderPath(type) {
            fetch(`/api/open_folder?type=${type}`);
        }

        function selectPtFile() {
            fetch('/api/select_pt_dialog')
                .then(res => res.json())
                .then(data => {
                    if (data.path) updateSelectedPt(data.path);
                });
        }

        function updateSelectedPt(path) {
            selectedPtPath.innerText = path;
        }

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.name.endsWith('.pt')) {
                    dropZone.classList.remove('error');
                    dropZone.innerText = 'Drop your .pt file';

                    const formData = new FormData();
                    formData.append('file', file);

                    fetch('/api/upload_pt', { method: 'POST', body: formData })
                        .then(res => res.json())
                        .then(data => {
                            if (data.success) updateSelectedPt(data.path);
                        });
                } else {
                    dropZone.classList.add('error');
                    dropZone.innerText = 'you select different format, please drop your .pt file';
                }
            }
        });

        // ⭐️ 최신 이미지 감지 및 자동 디스플레이 핵심 로직
        function refreshImageList() {
            fetch('/api/get_images')
                .then(res => res.json())
                .then(files => {
                    if (!files || files.length === 0) return;

                    const newestFile = files[0]; // 목록 최상단 (가장 최신 파일)

                    // 1) 새로운 예측 이미지가 도착했거나 최초 실행인 경우
                    if (newestFile !== currentTopFile) {
                        currentTopFile = newestFile;
                        selectedFile = newestFile; // 새로 들어온 최신 파일로 자동 선택 변경
                        displayImage(newestFile);  // 오른쪽 메인 화면 즉시 업데이트!
                    }

                    // 2) 좌측 목록 스크롤바 렌더링
                    imageListEl.innerHTML = '';
                    files.forEach((filename) => {
                        const li = document.createElement('li');
                        li.innerText = filename;
                        
                        if (filename === selectedFile) {
                            li.classList.add('active');
                        }

                        // 목록 아이템 클릭 시 선택 변경
                        li.onclick = () => {
                            selectedFile = filename;
                            displayImage(filename);
                            document.querySelectorAll('#image-list li').forEach(el => el.classList.remove('active'));
                            li.classList.add('active');
                        };
                        
                        imageListEl.appendChild(li);
                    });
                });
        }

        function displayImage(filename) {
            if (!filename) return;
            // 캐시 방지 타임스탬프(?t=...)
            viewerImg.src = `/predict_images/${encodeURIComponent(filename)}?t=${new Date().getTime()}`;
            viewerImg.style.display = 'block';
            viewerPlaceholder.style.display = 'none';
        }

        // 1초마다 주기적으로 새 예측 결과 확인
        setInterval(refreshImageList, 1000);
        refreshImageList();
    </script>
</body>
</html>
"""

# --- APIs ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/open_folder')
def open_folder():
    folder_type = request.args.get('type')
    target_dir = DIR_PREDICT if folder_type == 'image' else DIR_PT
    if sys.platform == 'win32':
        os.startfile(target_dir)
    return jsonify({"status": "ok"})

@app.route('/api/select_pt_dialog')
def select_pt_dialog():
    global selected_pt_file, loaded_model
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        initialdir=DIR_PT,
        title="Select .pt File",
        filetypes=[("PT files", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()

    if file_path:
        selected_pt_file = os.path.abspath(file_path)
        loaded_model = YOLO(selected_pt_file)
        return jsonify({"path": selected_pt_file})
    return jsonify({"path": selected_pt_file or "None"})

@app.route('/api/upload_pt', methods=['POST'])
def upload_pt():
    global selected_pt_file, loaded_model
    if 'file' not in request.files:
        return jsonify({"success": False}), 400
    
    file = request.files['file']
    if file.filename.endswith('.pt'):
        save_path = get_unique_filepath(DIR_PT, file.filename)
        file.save(save_path)
        
        selected_pt_file = os.path.abspath(save_path)
        loaded_model = YOLO(selected_pt_file)
        return jsonify({"success": True, "path": selected_pt_file})
    
    return jsonify({"success": False}), 400

# ⭐️ [수정] 임시 파일이나 예외 파일은 목록에서 완전히 제외
@app.route('/api/get_images')
def get_images():
    all_files = glob.glob(os.path.join(DIR_PREDICT, "*.jpg"))
    # _ 로 시작하는 임시 파일 제외
    valid_files = [f for f in all_files if not os.path.basename(f).startswith('_')]
    # 최신 수정 시간 순 정렬
    valid_files.sort(key=os.path.getmtime, reverse=True)
    file_names = [os.path.basename(f) for f in valid_files[:30]]
    return jsonify(file_names)

@app.route('/predict_images/<filename>')
def serve_image(filename):
    return send_from_directory(DIR_PREDICT, filename)

# ⭐️ [수정] 임시 파일 위치 변경 및 박스 개수 계산 안정화
@app.route('/predict', methods=['POST'])
def predict():
    global loaded_model, selected_pt_file
    
    if loaded_model is None:
        return jsonify({
            "error": "Model not selected", 
            "message": "서버에 선택된 .pt 모델이 없습니다. UI에서 .pt 파일을 지정해주세요."
        }), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    
    # ⭐️ 임시 파일은 predict_image 폴더가 아닌 pt_file 폴더에 저장하여 목록 노출 차단
    temp_path = os.path.join(DIR_PT, "_temp_input.jpg")
    file.save(temp_path)

    # YOLOv8 추론
    results = loaded_model(temp_path, conf=0.25)
    
    # 박스 개수 계산
    boxes = results[0].boxes
    detected_count = int(len(boxes)) if boxes is not None else 0

    print(f"[PREDICT] 추론 성공 | 감지된 사물 개수: {detected_count}개")

    # 결과 파일명 생성 (YYYYMMDD_HHMMSS_mmm.jpg)
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}.jpg"
    final_save_path = get_unique_filepath(DIR_PREDICT, time_str)

    # 박스가 그려진 이미지 최종 저장
    results[0].save(filename=final_save_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return jsonify({
        "status": "success",
        "detected_count": detected_count,
        "saved_filename": os.path.basename(final_save_path)
    })

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)