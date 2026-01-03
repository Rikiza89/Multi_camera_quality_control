"""
Flask-based Multi-Camera YOLO Quality Control System
Main application file: app.py
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import threading
import time
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class QualityControlState:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.cameras = {}  # {camera_id: {'thread': thread, 'cap': cap, 'role': 'product/package', 'index': int}}
        self.product_package_pairs = {}
        self.detection_buffers = {}  # {camera_id: [detections]}
        self.is_running = False
        self.check_interval = 1.0
        self.max_buffer_size = 100
        
        # Statistics
        self.total_checked = 0
        self.total_passed = 0
        self.total_failed = 0
        self.logs = []
        
        # Line stop detection
        self.line_stop_threshold = 5  # seconds without detection
        self.last_detection_time = {}  # {camera_id: timestamp}
        self.line_stopped = False
        
        # Lock for thread safety
        self.lock = threading.Lock()

state = QualityControlState()

# Camera thread
class CameraThread(threading.Thread):
    def __init__(self, camera_id, camera_index, role):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.role = role
        self.running = False
        self.cap = None
        
    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    if state.model_loaded and state.model:
                        self.process_frame(frame)
                    else:
                        # Just display frame without detection
                        _, buffer = cv2.imencode('.jpg', frame)
                        with state.lock:
                            if self.camera_id in state.cameras:
                                state.cameras[self.camera_id]['frame'] = buffer.tobytes()
                time.sleep(0.03)  # ~30 FPS
        except Exception as e:
            print(f"Camera thread error: {e}")
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
                
    def process_frame(self, frame):
        try:
            results = state.model(frame, verbose=False, conf=0.5)
            detected_classes = []
            
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = state.model.names[cls]
                    
                    detected_classes.append(class_name)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if self.role == 'product' else (255, 165, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Update detection buffer
            if detected_classes:
                with state.lock:
                    if self.camera_id not in state.detection_buffers:
                        state.detection_buffers[self.camera_id] = []
                    
                    timestamp = datetime.now()
                    state.detection_buffers[self.camera_id].append({
                        'classes': detected_classes,
                        'timestamp': timestamp.strftime("%H:%M:%S"),
                        'time_obj': timestamp
                    })
                    
                    # Update last detection time
                    state.last_detection_time[self.camera_id] = timestamp
                    
                    # Trim buffer
                    if len(state.detection_buffers[self.camera_id]) > state.max_buffer_size:
                        state.detection_buffers[self.camera_id].pop(0)
                
                # Emit detection to frontend
                try:
                    socketio.emit('detection', {
                        'camera_id': self.camera_id,
                        'role': self.role,
                        'classes': detected_classes
                    })
                except:
                    pass
            
            # Encode and store frame for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            with state.lock:
                if self.camera_id in state.cameras:
                    state.cameras[self.camera_id]['frame'] = buffer.tobytes()
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            traceback.print_exc()
            
    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

# Sequential checking thread
class SequentialChecker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        
    def run(self):
        self.running = True
        
        while self.running:
            try:
                time.sleep(state.check_interval)
                if state.is_running:
                    self.perform_check()
                    self.check_line_stop()
            except Exception as e:
                print(f"Checker error: {e}")
                traceback.print_exc()
                
    def perform_check(self):
        with state.lock:
            # Get product and package cameras
            product_cameras = [cid for cid, cam in state.cameras.items() if cam.get('role') == 'product']
            package_cameras = [cid for cid, cam in state.cameras.items() if cam.get('role') == 'package']
            
            if not product_cameras or not package_cameras:
                return
            
            # Check each product camera against package cameras
            for prod_cam in product_cameras:
                if prod_cam not in state.detection_buffers or not state.detection_buffers[prod_cam]:
                    continue
                    
                for pkg_cam in package_cameras:
                    if pkg_cam not in state.detection_buffers or not state.detection_buffers[pkg_cam]:
                        continue
                    
                    # Get oldest detections
                    product_detection = state.detection_buffers[prod_cam][0]
                    package_detection = state.detection_buffers[pkg_cam][0]
                    
                    # Remove from buffers
                    state.detection_buffers[prod_cam].pop(0)
                    state.detection_buffers[pkg_cam].pop(0)
                    
                    # Check pairing
                    self.check_pairing(product_detection, package_detection)
                    
    def check_pairing(self, product_detection, package_detection):
        product_classes = product_detection['classes']
        package_classes = package_detection['classes']
        
        for product in product_classes:
            if product in state.product_package_pairs:
                expected_package = state.product_package_pairs[product]
                
                state.total_checked += 1
                
                if expected_package in package_classes:
                    state.total_passed += 1
                    log_entry = {
                        'type': 'pass',
                        'timestamp': product_detection['timestamp'],
                        'message': f"✓ 合格: {product} ↔ {expected_package}"
                    }
                else:
                    state.total_failed += 1
                    log_entry = {
                        'type': 'fail',
                        'timestamp': product_detection['timestamp'],
                        'message': f"✗ 不合格: {product} と {', '.join(package_classes)} (期待: {expected_package})"
                    }
                
                state.logs.insert(0, log_entry)
                if len(state.logs) > 1000:
                    state.logs.pop()
                
                # Emit to frontend
                try:
                    socketio.emit('log_update', log_entry)
                    socketio.emit('stats_update', {
                        'total': state.total_checked,
                        'passed': state.total_passed,
                        'failed': state.total_failed
                    })
                except:
                    pass
                
    def check_line_stop(self):
        """Check if production line has stopped"""
        current_time = datetime.now()
        line_stopped = False
        stopped_cameras = []
        
        with state.lock:
            for camera_id, cam in state.cameras.items():
                if camera_id in state.last_detection_time:
                    time_diff = (current_time - state.last_detection_time[camera_id]).total_seconds()
                    if time_diff > state.line_stop_threshold:
                        line_stopped = True
                        stopped_cameras.append({
                            'id': camera_id,
                            'role': cam.get('role', 'unknown'),
                            'seconds': int(time_diff)
                        })
        
        if line_stopped != state.line_stopped:
            state.line_stopped = line_stopped
            try:
                socketio.emit('line_stop_alert', {
                    'stopped': line_stopped,
                    'cameras': stopped_cameras
                })
            except:
                pass
            
    def stop(self):
        self.running = False

checker_thread = None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_model', methods=['POST'])
def load_model():
    try:
        data = request.get_json()
        model_path = data.get('model_path', '')
        
        if not model_path:
            return jsonify({'success': False, 'error': 'モデルパスが指定されていません'})
            
        if not Path(model_path).exists():
            return jsonify({'success': False, 'error': f'モデルファイルが見つかりません: {model_path}'})
        
        from ultralytics import YOLO
        state.model = YOLO(model_path)
        state.model_loaded = True
        
        class_names = []
        if hasattr(state.model, 'names'):
            class_names = list(state.model.names.values())
        
        return jsonify({
            'success': True,
            'model_name': Path(model_path).name,
            'classes': class_names
        })
    except ImportError:
        return jsonify({'success': False, 'error': 'ultralyticsがインストールされていません。pip install ultralyticsを実行してください'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'モデル読み込みエラー: {str(e)}'})

@app.route('/api/add_camera', methods=['POST'])
def add_camera():
    try:
        data = request.get_json()
        camera_index = int(data.get('camera_index', 0))
        role = data.get('role', 'product')
        
        camera_id = f"cam_{len(state.cameras)}_{int(time.time())}"
        
        thread = CameraThread(camera_id, camera_index, role)
        
        with state.lock:
            state.cameras[camera_id] = {
                'thread': thread,
                'role': role,
                'index': camera_index,
                'frame': None
            }
        
        thread.start()
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'role': role,
            'index': camera_index
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'カメラ追加エラー: {str(e)}'})

@app.route('/api/remove_camera/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    try:
        with state.lock:
            if camera_id in state.cameras:
                thread = state.cameras[camera_id]['thread']
                thread.stop()
                thread.join(timeout=2)
                del state.cameras[camera_id]
                
                if camera_id in state.detection_buffers:
                    del state.detection_buffers[camera_id]
                if camera_id in state.last_detection_time:
                    del state.last_detection_time[camera_id]
                
                return jsonify({'success': True})
            return jsonify({'success': False, 'error': 'カメラが見つかりません'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/video_feed/<camera_id>')
def video_feed(camera_id):
    def generate():
        while True:
            try:
                with state.lock:
                    if camera_id in state.cameras and state.cameras[camera_id].get('frame'):
                        frame = state.cameras[camera_id]['frame']
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)
            except Exception as e:
                print(f"Video feed error: {e}")
                break
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_system():
    global checker_thread
    
    try:
        if not state.model_loaded:
            return jsonify({'success': False, 'error': 'モデルが読み込まれていません'})
        
        if not state.product_package_pairs:
            return jsonify({'success': False, 'error': '製品-パッケージペアが定義されていません'})
        
        with state.lock:
            state.is_running = True
        
        # Start checker thread
        if checker_thread is None or not checker_thread.is_alive():
            checker_thread = SequentialChecker()
            checker_thread.start()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    try:
        with state.lock:
            state.is_running = False
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_pair', methods=['POST'])
def add_pair():
    try:
        data = request.get_json()
        product = data.get('product', '').strip()
        package = data.get('package', '').strip()
        
        if not product or not package:
            return jsonify({'success': False, 'error': '製品名とパッケージ名を入力してください'})
        
        with state.lock:
            state.product_package_pairs[product] = package
        save_config()
        
        return jsonify({'success': True, 'pairs': state.product_package_pairs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/remove_pair/<product>', methods=['DELETE'])
def remove_pair(product):
    try:
        with state.lock:
            if product in state.product_package_pairs:
                del state.product_package_pairs[product]
        save_config()
        return jsonify({'success': True, 'pairs': state.product_package_pairs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_pairs', methods=['GET'])
def get_pairs():
    return jsonify({'pairs': state.product_package_pairs})

@app.route('/api/get_cameras', methods=['GET'])
def get_cameras():
    cameras_info = []
    with state.lock:
        for camera_id, cam in state.cameras.items():
            cameras_info.append({
                'id': camera_id,
                'role': cam.get('role', 'unknown'),
                'index': cam.get('index', 0)
            })
    return jsonify({'cameras': cameras_info})

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    return jsonify({
        'total': state.total_checked,
        'passed': state.total_passed,
        'failed': state.total_failed
    })

@app.route('/api/reset_stats', methods=['POST'])
def reset_stats():
    with state.lock:
        state.total_checked = 0
        state.total_passed = 0
        state.total_failed = 0
        state.logs.clear()
    return jsonify({'success': True})

@app.route('/api/get_logs', methods=['GET'])
def get_logs():
    return jsonify({'logs': state.logs[:100]})

@app.route('/api/set_line_stop_threshold', methods=['POST'])
def set_line_stop_threshold():
    try:
        data = request.get_json()
        threshold = int(data.get('threshold', 5))
        state.line_stop_threshold = threshold
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def save_config():
    try:
        with open('qc_config.json', 'w', encoding='utf-8') as f:
            json.dump(state.product_package_pairs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"設定保存エラー: {e}")

def load_config():
    try:
        if Path('qc_config.json').exists():
            with open('qc_config.json', 'r', encoding='utf-8') as f:
                state.product_package_pairs = json.load(f)
    except Exception as e:
        print(f"設定読み込みエラー: {e}")

if __name__ == '__main__':
    load_config()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
