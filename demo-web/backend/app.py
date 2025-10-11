from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import base64
import sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import cv2
import threading
import time

# 建議安裝 psutil 以取得 RAM 用量：pip install psutil
try:
    import psutil
except Exception:
    psutil = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.init_prune_model import init_prune_model, detection

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 原有全域變量
net = None
img_normalization = None
box_coder = None
channel_information_path = "../prune_channel_information.csv"
class_images = []
inference_count = 0
param_count = 0

# 新增影片處理狀態
video_processing_status = {
    'is_processing': False,
    'progress': 0,
    'total_frames': 0,
    'current_frame': 0,
    'result': None,
    'error': None,
    'current_detection': None
}

def get_class_color(class_name):
    """根據類別名稱返回對應顏色"""
    color_map = {
        'person': 'red',
        'car': 'blue', 
        'truck': 'green',
        'bus': 'yellow',
        'motorcycle': 'purple',
        'bicycle': 'orange',
        'dog': 'pink',
        'cat': 'cyan',
        'unknown': 'red'
    }
    return color_map.get(class_name, 'red')

def get_process_memory_mb():
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        return int(mem_bytes / (1024 * 1024))
    except Exception:
        return None

def get_gpu_status():
    if not torch.cuda.is_available():
        return None
    try:
        device_index = 0
        props = torch.cuda.get_device_properties(device_index)
        total_mb = int(props.total_memory / (1024 * 1024))
        used_mb = int(torch.cuda.memory_allocated(device_index) / (1024 * 1024))
        return {
            "name": props.name,
            "used_MB": used_mb,
            "total_MB": total_mb
        }
    except Exception:
        return None

# 原有路由保持不變
@app.route('/model_status', methods=['GET'])
def model_status():
    global inference_count, param_count, net
    
    try:
        import psutil
        mem_mb = int(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except:
        mem_mb = 0
    
    device = None
    try:
        if net is not None:
            device = str(next(net.parameters()).device)
    except Exception:
        device = None
    
    return jsonify({
        "memory_usage_MB": mem_mb,
        "inference_count": inference_count,
        "param_count": param_count,
        "gpu": None,
        "device": device
    })

@app.route('/upload_pth', methods=['POST'])
def upload_pth():
    global net, img_normalization, box_coder, param_count
    
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(file_path)
    
    try:
        net, img_normalization, box_coder = init_prune_model(file_path, channel_information_path=channel_information_path)
        try:
            param_count = sum(p.numel() for p in net.parameters())
        except Exception:
            param_count = 0
        
        return jsonify({"result": f"模型已成功載入, {file_path}", "param_count": param_count})
    except Exception as e:
        net, img_normalization, box_coder = None, None, None
        param_count = 0
        return jsonify({"error": str(e)}), 500

@app.route('/upload_channel_file', methods=['POST'])
def upload_channel_file():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(file_path)
    
    try:
        global channel_information_path
        channel_information_path = file_path
        return jsonify({"result": f"Channel importance file uploaded successfully: {file_path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global inference_count
    
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(file_path)
    
    out_img_pil, result = analyze_image_with_model_input(file_path)
    
    # 僅在沒有 error 時才視為一次成功推論
    if isinstance(result, dict) and ("error" not in result):
        inference_count += 1
    
    buf = io.BytesIO()
    out_img_pil.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({"result": result, "image": img_b64})

@app.route('/upload_class_image', methods=['POST'])
def upload_class_image():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, f.filename)
    if not os.path.exists(file_path):
        f.save(file_path)
    
    if file_path not in class_images:
        analyze_image_with_model_class(file_path)
    
    result = class_images
    results = []
    for idx, res in enumerate(result):
        results.append({"id": idx, "path": res})
    
    return jsonify({"result": results})

@app.route('/clear_camera_cache', methods=['POST'])
def clear_camera_cache():
    global class_images, inference_count, video_processing_status
    
    class_images = []
    inference_count = 0
    
    # 重置影片處理狀態
    video_processing_status = {
        'is_processing': False,
        'progress': 0,
        'total_frames': 0,
        'current_frame': 0,
        'result': None,
        'error': None,
        'current_detection': None
    }
    
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov')):
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, fname))
            except Exception:
                pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    return jsonify({"result": "已清空攝影機相關快取/暫存影像/分析狀態"})

# 新增影片處理路由
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_processing_status
    
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "No video file"}), 400
    
    # 檢查是否正在處理其他影片
    if video_processing_status['is_processing']:
        return jsonify({"error": "另一個影片正在處理中，請稍後再試"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(file_path)
    
    # 重置處理狀態
    video_processing_status = {
        'is_processing': True,
        'progress': 0,
        'total_frames': 0,
        'current_frame': 0,
        'result': None,
        'error': None,
        'current_detection': None
    }
    
    # 在背景執行緒中處理影片
    thread = threading.Thread(target=process_video_async, args=(file_path,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "message": "影片上傳成功，開始處理",
        "filename": f.filename,
        "status": "processing"
    })

def process_video_async(video_path):
    global video_processing_status, net, img_normalization, box_coder, class_images, inference_count
    
    try:
        # 檢查模型和類別圖像是否已載入
        if net is None or img_normalization is None or box_coder is None:
            video_processing_status['error'] = "模型尚未載入"
            video_processing_status['is_processing'] = False
            return
        
        if len(class_images) == 0:
            video_processing_status['error'] = "尚未上傳任何類別圖像"
            video_processing_status['is_processing'] = False
            return
        
        # 開啟影片
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            video_processing_status['error'] = "無法開啟影片檔案"
            video_processing_status['is_processing'] = False
            return
        
        # 獲取影片資訊
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"原始影片資訊: {width}x{height}, {fps}fps, {total_frames}frames")
        
        video_processing_status['total_frames'] = total_frames
        
        # 建立可靠的VideoWriter創建函數
        def create_reliable_video_writer(base_path, w, h, fps_val):
            # 優先嘗試的編碼器組合
            configs = [
                ('mp4v', 'mp4'),   # 最穩定的 MPEG-4
                ('XVID', 'avi'),   # AVI 常見
                ('MJPG', 'avi'),   # MJPEG
                ('DIVX', 'avi'),   # DivX
                ('H264', 'mp4'),   # H.264 (需安裝正確 DLL)
                ('X264', 'mp4'),   # 同上，某些環境用這個
            ]

            for codec, ext in configs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    output_path = f"{base_path}_{codec}.{ext}"

                    print(f"嘗試創建VideoWriter: {codec} -> {output_path}")
                    writer = cv2.VideoWriter(output_path, fourcc, fps_val, (w, h))

                    if writer.isOpened():
                        # 測試寫入一幀黑畫面
                        test_frame = np.zeros((h, w, 3), dtype=np.uint8)
                        writer.write(test_frame)
                        writer.release()

                        # 確認檔案大小 > 0
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                            print(f"✅ 成功創建可用的VideoWriter: {codec}")
                            # 重新開一個真正用來輸出的 writer
                            writer = cv2.VideoWriter(output_path, fourcc, fps_val, (w, h))
                            return writer, output_path, codec
                        else:
                            print(f"❌ VideoWriter寫入測試檔案失敗: {codec}")
                            if os.path.exists(output_path):
                                os.remove(output_path)
                    else:
                        print(f"❌ VideoWriter無法開啟: {codec}")
                        writer.release()

                except Exception as e:
                    print(f"❌ 創建VideoWriter時發生異常: {codec} - {str(e)}")
                    continue

            return None, None, None

        
        # 創建VideoWriter
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_base = os.path.join(UPLOAD_FOLDER, f"detected_{base_name}")
        
        out, final_output_path, used_codec = create_reliable_video_writer(output_base, width, height, fps)
        
        if out is None:
            video_processing_status['error'] = "無法創建任何可用的VideoWriter，請檢查OpenCV安裝和編碼器支援"
            video_processing_status['is_processing'] = False
            return
        
        detection_count = 0
        detection_stats = {}
        frame_count = 0
        
        print(f"開始處理影片，使用編碼器: {used_codec}")
        print(f"輸出路徑: {final_output_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 檢查是否被取消
            if not video_processing_status['is_processing']:
                break
            
            frame_count += 1
            video_processing_status['current_frame'] = frame_count
            video_processing_status['progress'] = (frame_count / total_frames) * 100
            
            # 確保原始幀格式正確
            if frame.shape != (height, width, 3):
                frame = cv2.resize(frame, (width, height))
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # 暫存當前幀為圖像檔案
            temp_frame_path = os.path.join(UPLOAD_FOLDER, "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # 使用現有的檢測函數
                detected_frame, result = detection(
                    net, box_coder, img_normalization,
                    temp_frame_path, class_images
                )
                
                # 處理檢測結果圖像
                if detected_frame is not None and detected_frame.size > 0:
                    # 確保數據類型
                    if detected_frame.dtype != np.uint8:
                        if detected_frame.max() <= 1.0:
                            detected_frame = (detected_frame * 255).astype(np.uint8)
                        else:
                            detected_frame = detected_frame.astype(np.uint8)
                    
                    # 如果是RGB，轉換為BGR（matplotlib輸出通常是RGB）
                    if len(detected_frame.shape) == 3 and detected_frame.shape[2] == 3:
                        # 檢查是否需要轉換（簡單的顏色檢測）
                        detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)
                    
                    # 強制resize到正確尺寸
                    if detected_frame.shape[:2] != (height, width):
                        detected_frame = cv2.resize(detected_frame, (width, height))
                    
                    # 確保內存連續
                    if not detected_frame.flags['C_CONTIGUOUS']:
                        detected_frame = np.ascontiguousarray(detected_frame)
                    
                    frame_to_write = detected_frame
                else:
                    frame_to_write = frame
                
                # 格式化檢測結果
                boxes = []
                label_names = []
                scores = []
                colors = []
                polygons = []
                
                if isinstance(result, dict) and 'detections' in result:
                    detections = result['detections']
                    detection_count += len(detections)
                    
                    for detection_item in detections:
                        if 'bbox' in detection_item:
                            bbox = detection_item['bbox']
                            boxes.append([float(coord) for coord in bbox])
                        
                        if 'class' in detection_item:
                            class_name = detection_item['class']
                            label_names.append(str(class_name))
                            detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                        
                        if 'confidence' in detection_item or 'score' in detection_item:
                            score = detection_item.get('confidence', detection_item.get('score', 0.0))
                            scores.append(float(score))
                        
                        color = get_class_color(class_name if 'class' in detection_item else 'unknown')
                        colors.append(color)
                        
                        if 'polygon' in detection_item:
                            polygon = detection_item['polygon']
                            polygons.append([float(coord) for coord in polygon])
                
                formatted_result = {
                    "boxes": boxes,
                    "label_names": label_names,
                    "scores": scores,
                    "colors": colors,
                    "image_id": f"frame_{frame_count}",
                    "polygons": polygons if polygons else None,
                    "detections": result.get('detections', []) if isinstance(result, dict) else []
                }
                
                # 寫入處理後的幀
                try:
                    out.write(frame_to_write)
                except Exception as e:
                    print(f"寫入第 {frame_count} 幀發生錯誤: {e}")
                # 更新前端顯示
                update_interval = max(1, fps * 3)
                # update_interval = max(1, 15) # 每 15 偵
                if frame_count % update_interval == 0 or frame_count % 30 == 0:
                    try:
                        display_frame = frame_to_write.copy()
                        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
                            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        
                        pil_detected = Image.fromarray(display_frame)
                        buf = io.BytesIO()
                        pil_detected.save(buf, format='PNG')
                        detected_img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        
                        video_processing_status['current_detection'] = {
                            'image': detected_img_b64,
                            'frame_number': frame_count,
                            'result': formatted_result,
                            'timestamp': f"{frame_count / fps:.1f}s"
                        }
                    except Exception as e:
                        print(f"更新前端顯示失敗: {str(e)}")
                
                inference_count += 1
                
            except Exception as e:
                print(f"處理第 {frame_count} 幀時發生錯誤: {str(e)}")
                out.write(frame)
            
            # 清理暫存檔案
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            if frame_count % 30 == 0:
                print(f"處理進度: {frame_count}/{total_frames} ({video_processing_status['progress']:.1f}%)")
        
        # 釋放資源
        cap.release()
        out.release()
        
        # 檢查是否被取消
        if not video_processing_status['is_processing']:
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            return
        
        print(f"影片處理完成，檢測到 {detection_count} 個物件")
        print(f"已儲存檢測後影片: {final_output_path}")
        
        # 檢查最終檔案
        if os.path.exists(final_output_path):
            file_size = os.path.getsize(final_output_path)
            print(f"輸出檔案大小: {file_size / (1024*1024):.2f} MB")
            
            video_b64 = None
            if file_size > 1000:
                try:
                    with open(final_output_path, 'rb') as video_file:
                        video_b64 = base64.b64encode(video_file.read()).decode('utf-8')
                except Exception as e:
                    print(f"轉換base64失敗: {str(e)}")
        else:
            file_size = 0
            video_b64 = None
        
        # 設定處理結果
        video_processing_status['result'] = {
            'output_video': video_b64,
            'output_file_path': final_output_path,
            'output_filename': os.path.basename(final_output_path),
            'file_size_mb': file_size / (1024*1024) if file_size > 0 else 0,
            'total_detections': detection_count,
            'total_frames': total_frames,
            'detection_stats': detection_stats,
            'processed_frames': frame_count,
            'codec_used': used_codec,
            'message': f"影片處理完成！使用編碼器 {used_codec}，檢測到 {detection_count} 個物件，處理了 {frame_count} 幀影像。已儲存至 {final_output_path}"
        }
        
    except Exception as e:
        print(f"影片處理發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        video_processing_status['error'] = f"影片處理失敗: {str(e)}"
    finally:
        video_processing_status['is_processing'] = False
        video_processing_status['progress'] = 100


@app.route('/video_progress', methods=['GET'])
def video_progress():
    global video_processing_status
    return jsonify({
        'is_processing': video_processing_status['is_processing'],
        'progress': video_processing_status['progress'],
        'current_frame': video_processing_status['current_frame'],
        'total_frames': video_processing_status['total_frames'],
        'completed': not video_processing_status['is_processing'] and video_processing_status['progress'] >= 100,
        'error': video_processing_status['error']
    })

@app.route('/video_result', methods=['GET'])
def video_result():
    global video_processing_status
    if video_processing_status['result']:
        return jsonify(video_processing_status['result'])
    elif video_processing_status['error']:
        return jsonify({'error': video_processing_status['error']}), 500
    else:
        return jsonify({'error': '沒有可用的結果'}), 404

@app.route('/cancel_video_processing', methods=['POST'])
def cancel_video_processing():
    global video_processing_status
    if video_processing_status['is_processing']:
        video_processing_status['is_processing'] = False
        video_processing_status['error'] = '處理已被使用者取消'
        return jsonify({'message': '影片處理已取消'})
    else:
        return jsonify({'message': '沒有正在進行的影片處理'})

# 新增API獲取當前檢測幀
@app.route('/current_detection_frame', methods=['GET'])
def current_detection_frame():
    global video_processing_status
    if video_processing_status.get('current_detection'):
        return jsonify(video_processing_status['current_detection'])
    else:
        return jsonify({'error': '無當前檢測幀'}), 404

# 新增下載API
@app.route('/download_processed_video', methods=['GET'])
def download_processed_video():
    global video_processing_status
    
    if video_processing_status.get('result') and video_processing_status['result'].get('output_file_path'):
        file_path = video_processing_status['result']['output_file_path']
        filename = video_processing_status['result']['output_filename']
        
        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name=filename,
                mimetype='video/mp4'
            )
        else:
            return jsonify({'error': '處理後的影片檔案不存在'}), 404
    else:
        return jsonify({'error': '沒有可下載的影片'}), 404

@app.route('/list_processed_videos', methods=['GET'])
def list_processed_videos():
    processed_videos = []
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.startswith('detected_') and filename.endswith('.mp4'):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file_size = os.path.getsize(file_path)
            file_time = os.path.getctime(file_path)
            
            processed_videos.append({
                'filename': filename,
                'size_mb': file_size / (1024*1024),
                'created_time': time.ctime(file_time),
                'download_url': f'/download_video/{filename}'
            })
    
    return jsonify({'videos': processed_videos})

@app.route('/download_video/<filename>', methods=['GET'])
def download_video_by_name(filename):
    # 安全性檢查：只允許下載detected_開頭的mp4檔案
    if not filename.startswith('detected_') or not filename.endswith('.mp4'):
        return jsonify({'error': '檔案名稱不合法'}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if os.path.exists(file_path):
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
    else:
        return jsonify({'error': '檔案不存在'}), 404

# 原有函數保持不變
def analyze_image_with_model_input(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    
    try:
        if len(class_images) == 0:
            return pil_img, {"error": "尚未上傳任何類別圖像"}
        
        global net, img_normalization, box_coder
        if net is None or img_normalization is None or box_coder is None:
            return pil_img, {"error": "模型尚未載入"}
        
        bgr_img, result = detection(net, box_coder, img_normalization, img_path, class_images)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        return pil_img, result
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return pil_img, {"error": f"推論失敗: {str(e)}"}

def analyze_image_with_model_class(img_path):
    class_images.append(img_path)
    return class_images

if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=True)
