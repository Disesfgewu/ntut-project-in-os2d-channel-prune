import React, { useRef, useEffect, useState } from "react";
import "./App.css";

// 條件性導入 Chart 組件
let Line, Doughnut, Chart;
try {
  const chartImports = require("react-chartjs-2");
  Line = chartImports.Line;
  Doughnut = chartImports.Doughnut;
  
  const chartjsImports = require("chart.js");
  Chart = chartjsImports.Chart;
  
  // 註冊必要組件
  Chart.register(
    chartjsImports.CategoryScale,
    chartjsImports.LinearScale,
    chartjsImports.PointElement,
    chartjsImports.LineElement,
    chartjsImports.Tooltip,
    chartjsImports.Legend,
    chartjsImports.ArcElement
  );
} catch (error) {
  console.warn("Chart.js components not available:", error);
  Line = null;
  Doughnut = null;
}

function StatusPanel({ isStreaming }) {
  const [modelStatus, setModelStatus] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    let intervalId;
    if (isStreaming) {
      const fetchStatus = async () => {
        try {
          const res = await fetch("http://localhost:5000/model_status");
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          setModelStatus(data);
          setHistory((prev) => [...prev.slice(-19), data]);
        } catch (e) {
          console.error("Failed to fetch model status:", e);
          // 保持上一次的狀態，不設為 null
        }
      };
      fetchStatus();
      intervalId = setInterval(fetchStatus, 5000);
    }
    return () => intervalId && clearInterval(intervalId);
  }, [isStreaming]);

  // 不在串流時不顯示
  if (!isStreaming) return null;

  // Loading 狀態
  if (!modelStatus) {
    return (
      <div className="panel grid-bottom-channel">
        <h3>模型動態參數監控</h3>
        <div>Loading...</div>
      </div>
    );
  }

  // 如果 Chart 組件不可用，顯示純文字版本
  if (!Line || !Doughnut) {
    return (
      <div className="panel grid-bottom-channel">
        <h3>模型動態參數監控</h3>
        <div>記憶體用量: {modelStatus.memory_usage_MB || 0} MB</div>
        <div>推論次數: {modelStatus.inference_count || 0}</div>
        <div>參數數量: {modelStatus.param_count || 0}</div>
        <div style={{ color: 'red', fontSize: '12px', marginTop: '8px' }}>
          圖表組件載入失敗，請檢查 react-chartjs-2 安裝
        </div>
      </div>
    );
  }

  // 準備圖表數據，確保數據有效
  const validHistory = history.filter(h => h && typeof h === 'object');
  const hasValidData = validHistory.length > 0;

  if (!hasValidData) {
    return (
      <div className="panel grid-bottom-channel">
        <h3>模型動態參數監控</h3>
        <div>等待資料...</div>
      </div>
    );
  }

  const memoryData = {
    labels: validHistory.map((_, idx) => `${idx * 5}s`),
    datasets: [{
      data: validHistory.map(h => h.memory_usage_MB || 0),
      label: "記憶體(MB)",
      fill: false,
      borderColor: "#42a5f5",
      tension: 0.25
    }]
  };

  const inferenceData = {
    labels: validHistory.map((_, idx) => `${idx * 5}s`),
    datasets: [{
      data: validHistory.map(h => h.inference_count || 0),
      label: "推論次數",
      fill: false,
      borderColor: "#fb8c00",
      tension: 0.25
    }]
  };

  const paramCnt = modelStatus.param_count || 0;
  const paramData = {
    labels: ["參數數量", "其它"],
    datasets: [{
      data: [paramCnt, Math.max(1, 1e7 - paramCnt)],
      backgroundColor: ["#43a047", "#ccc"]
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { ticks: { maxTicksLimit: 8 } } }
  };

  return (
    <div className="panel grid-bottom-channel">
      <h3>模型動態參數監控</h3>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        <div style={{ minWidth: 200, flex: 1, height: 200 }}>
          <div style={{ fontSize: 14, marginBottom: 8 }}>
            記憶體用量: {modelStatus.memory_usage_MB || 0} MB
          </div>
          <div style={{ height: 150 }}>
            <Line data={memoryData} options={chartOptions} />
          </div>
        </div>
        <div style={{ minWidth: 200, flex: 1, height: 200 }}>
          <div style={{ fontSize: 14, marginBottom: 8 }}>
            推論次數: {modelStatus.inference_count || 0}
          </div>
          <div style={{ height: 150 }}>
            <Line data={inferenceData} options={chartOptions} />
          </div>
        </div>
        <div style={{ minWidth: 160, flex: "0 0 200px", height: 200 }}>
          <div style={{ fontSize: 14, marginBottom: 8 }}>
            參數數量: {paramCnt}
          </div>
          <div style={{ height: 150 }}>
            <Doughnut 
              data={paramData} 
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
              }} 
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [uploadedPth, setUploadedPth] = useState(null);
  const [pthResponse, setPthResponse] = useState(null);
  const [uploadedInputImg, setUploadedInputImg] = useState(null);
  const [inputImgResponse, setInputImgResponse] = useState(null);
  const [uploadedClassImg, setUploadedClassImg] = useState(null);
  const [classImgResponse, setClassImgResponse] = useState(null);
  const [uploadedChannelFile, setUploadedChannelFile] = useState(null);
  const [channelFileResponse, setChannelFileResponse] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const streamInterval = useRef(null);

  // 新增影片處理相關狀態
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [inputVideoResponse, setInputVideoResponse] = useState(null);
  const [isVideoProcessing, setIsVideoProcessing] = useState(false);
  const [videoProcessingInterval, setVideoProcessingInterval] = useState(null);
  const [videoProgress, setVideoProgress] = useState(0);

  const handlePthChange = (e) => setUploadedPth(e.target.files[0]);
  const handlePthUpload = async () => {
    if (!uploadedPth) return alert("請選擇 .pth 檔案");
    const formData = new FormData();
    formData.append("file", uploadedPth);
    try {
      const res = await fetch("http://localhost:5000/upload_pth", { 
        method: "POST", 
        body: formData 
      });
      const data = await res.json();
      setPthResponse(data);
    } catch (e) {
      console.error("Upload pth failed:", e);
      alert("上傳失敗");
    }
  };

  const handleInputImgChange = (e) => setUploadedInputImg(e.target.files[0]);
  const handleInputImgUpload = async () => {
    if (!uploadedInputImg) return alert("請選擇 input 圖片");
    const formData = new FormData();
    formData.append("image", uploadedInputImg);
    try {
      const res = await fetch("http://localhost:5000/upload_image", { 
        method: "POST", 
        body: formData 
      });
      const data = await res.json();
      setInputImgResponse(data);
      if (data.image) setOutputImage(data.image);
    } catch (e) {
      console.error("Upload image failed:", e);
    }
  };

  const handleClassImgChange = (e) => setUploadedClassImg(e.target.files[0]);
  const handleClassImgUpload = async () => {
    if (!uploadedClassImg) return alert("請選擇 class 圖片");
    const formData = new FormData();
    formData.append("image", uploadedClassImg);
    try {
      const res = await fetch("http://localhost:5000/upload_class_image", { 
        method: "POST", 
        body: formData 
      });
      const data = await res.json();
      setClassImgResponse(data);
      if (data.image) setOutputImage(data.image);
    } catch (e) {
      console.error("Upload class image failed:", e);
    }
  };

  const handleChannelFileChange = (e) => setUploadedChannelFile(e.target.files[0]);
  const handleChannelFileUpload = async () => {
    if (!uploadedChannelFile) return alert("請選擇 Channel Importance 檔案");
    const formData = new FormData();
    formData.append("file", uploadedChannelFile);
    try {
      const res = await fetch("http://localhost:5000/upload_channel_file", { 
        method: "POST", 
        body: formData 
      });
      const data = await res.json();
      setChannelFileResponse(data);
    } catch (e) {
      console.error("Upload channel file failed:", e);
    }
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("你的瀏覽器不支援攝影機串流");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const videoEl = videoRef.current;
      if (!videoEl) {
        console.error("Video element not available");
        return;
      }
      videoEl.srcObject = stream;
      videoEl.onloadedmetadata = () => {
        if (videoEl && typeof videoEl.play === "function") {
          videoEl.play().catch(err => {
            console.error("Failed to play video:", err);
          });
        }
      };
      setIsStreaming(true);
    } catch (err) {
      console.error("Camera access failed:", err);
      alert("無法取得攝影機權限: " + err.message);
    }
  };

  const stopCamera = () => {
    setIsStreaming(false);
    if (streamInterval.current) {
      clearInterval(streamInterval.current);
      streamInterval.current = null;
    }
    const videoEl = videoRef.current;
    if (videoEl) {
      videoEl.onloadedmetadata = null;
      if (videoEl.srcObject) {
        videoEl.srcObject.getTracks().forEach(track => track.stop());
        videoEl.srcObject = null;
      }
    }
  };

  useEffect(() => {
    if (isStreaming) {
      streamInterval.current = setInterval(async () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video || video.videoWidth === 0 || video.videoHeight === 0) return;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(async (blob) => {
          if (!blob) return;
          const formData = new FormData();
          formData.append("image", blob, "frame.png");
          try {
            const res = await fetch("http://localhost:5000/upload_image", { 
              method: "POST", 
              body: formData 
            });
            const data = await res.json();
            if (data.image) setOutputImage(data.image);
          } catch (e) {
            console.error("Stream upload failed:", e);
          }
        }, "image/png");
      }, 2000);
      
      return () => {
        if (streamInterval.current) {
          clearInterval(streamInterval.current);
        }
      };
    } else {
      if (streamInterval.current) {
        clearInterval(streamInterval.current);
        streamInterval.current = null;
      }
    }
  }, [isStreaming]);

  const clearCameraImage = async () => {
    try {
      await fetch("http://localhost:5000/clear_camera_cache", { method: "POST" });
      setOutputImage(null);
      setInputVideoResponse(null);
      setIsVideoProcessing(false);
      setVideoProgress(0);
      if (videoProcessingInterval) {
        clearInterval(videoProcessingInterval);
        setVideoProcessingInterval(null);
      }
    } catch (e) {
      console.error("Clear cache failed:", e);
    }
  };

  // 影片處理相關函數
  const handleInputVideoChange = (e) => setUploadedVideo(e.target.files[0]);
  
  const handleInputVideoUpload = async () => {
    if (!uploadedVideo) return alert("請選擇影片檔案");
    
    const formData = new FormData();
    formData.append("video", uploadedVideo);
    
    setIsVideoProcessing(true);
    setInputVideoResponse(null);
    setVideoProgress(0);
    
    try {
      const res = await fetch("http://localhost:5000/upload_video", { 
        method: "POST", 
        body: formData 
      });
      const data = await res.json();
      
      if (data.error) {
        setInputVideoResponse({ error: data.error });
        setIsVideoProcessing(false);
      } else {
        setInputVideoResponse(data);
        // 開始監控影片處理和當前檢測幀
        startVideoProcessingMonitor();
      }
      
    } catch (e) {
      console.error("Upload video failed:", e);
      alert("上傳影片失敗");
      setIsVideoProcessing(false);
    }
  };

  // 新增監控函數
  const startVideoProcessingMonitor = () => {
    const interval = setInterval(async () => {
      try {
        // 檢查處理進度
        const progressRes = await fetch("http://localhost:5000/video_progress");
        const progressData = await progressRes.json();
        
        setVideoProgress(progressData.progress || 0);
        
        if (progressData.completed) {
          // 處理完成
          setIsVideoProcessing(false);
          clearInterval(interval);
          setVideoProcessingInterval(null);
          
          // 獲取最終結果
          const resultRes = await fetch("http://localhost:5000/video_result");
          const resultData = await resultRes.json();
          setInputVideoResponse(resultData);
          
        } else if (progressData.error) {
          // 處理出錯
          setIsVideoProcessing(false);
          clearInterval(interval);
          setVideoProcessingInterval(null);
          setInputVideoResponse({ error: progressData.error });
          
        } else if (progressData.is_processing) {
          // 正在處理中，獲取當前檢測幀
          try {
            const currentFrameRes = await fetch("http://localhost:5000/current_detection_frame");
            if (currentFrameRes.ok) {
              const currentFrameData = await currentFrameRes.json();
              // 更新輸出圖像為當前檢測幀
              setOutputImage(currentFrameData.image);
            }
          } catch (frameError) {
            // 當前沒有檢測幀，不做處理
            console.log("No current detection frame available");
          }
        }
        
      } catch (error) {
        console.error("Monitor video processing failed:", error);
      }
    }, 2000); // 每2秒檢查一次
    
    setVideoProcessingInterval(interval);
  };

  // 清理函數
  useEffect(() => {
    return () => {
      if (videoProcessingInterval) {
        clearInterval(videoProcessingInterval);
      }
    };
  }, [videoProcessingInterval]);

  return (
    <div className="grid-container">
      <div className="panel grid-top-left">
        <h3>上傳 .pth 檔案</h3>
        <input type="file" accept=".pth" onChange={handlePthChange} />
        <button onClick={handlePthUpload}>上傳 .pth</button>
        <pre>{pthResponse && JSON.stringify(pthResponse, null, 2)}</pre>
      </div>
      
      <div className="panel grid-top-right">
        <h3>分析後輸出影像</h3>
        {outputImage ? (
          <img
            src={outputImage.startsWith("data:") ? outputImage : `data:image/png;base64,${outputImage}`}
            alt="Output"
            style={{ maxWidth: 400 }}
          />
        ) : (
          <p>尚無輸出影像</p>
        )}
        <div style={{ marginTop: 16 }}>
          <button onClick={startCamera} disabled={isStreaming || isVideoProcessing}>
            開始串流
          </button>
          <button onClick={stopCamera} disabled={!isStreaming}>
            停止串流
          </button>
          <button onClick={clearCameraImage} style={{ marginLeft: 8 }}>
            清空攝影機分析影像
          </button>
        </div>
        {isVideoProcessing && (
          <div style={{ marginTop: 10, color: 'blue', fontSize: '14px' }}>
            📹 正在處理影片 ({Math.round(videoProgress)}%)，檢測結果即時顯示中...
          </div>
        )}
      </div>
      
      <div className="panel grid-center-camera">
        <h3>攝影機即時預覽</h3>
        <video
          ref={videoRef}
          style={{ width: 320, maxHeight: 240, background: "#000" }}
          autoPlay
          muted
          playsInline
        />
      </div>
      
      <div className="panel grid-bottom-left">
        <h3>上傳 input 圖片</h3>
        <input type="file" accept="image/*" onChange={handleInputImgChange} />
        <button onClick={handleInputImgUpload}>上傳圖片</button>
        <pre>{inputImgResponse && JSON.stringify(inputImgResponse, null, 2)}</pre>
      </div>

      <div className="panel grid-bottom-middle">
        <h3>上傳 input mp4 file</h3>
        <input 
          type="file" 
          accept="video/mp4,video/avi,video/mov" 
          onChange={handleInputVideoChange}
          disabled={isVideoProcessing}
        />
        <button 
          onClick={handleInputVideoUpload}
          disabled={isVideoProcessing}
        >
          {isVideoProcessing ? "處理中..." : "上傳影片"}
        </button>
        
        {isVideoProcessing && (
          <div style={{ marginTop: 10 }}>
            <div style={{ color: 'blue', marginBottom: 5 }}>
              📹 正在處理影片中...
            </div>
            <div style={{
              width: '100%',
              height: '20px',
              backgroundColor: '#e0e0e0',
              borderRadius: '10px',
              overflow: 'hidden'
            }}>
              <div style={{
                height: '100%',
                backgroundColor: '#4CAF50',
                width: `${videoProgress}%`,
                transition: 'width 0.3s ease'
              }}></div>
            </div>
            <div style={{ fontSize: '12px', marginTop: 5 }}>
              進度: {Math.round(videoProgress)}%
            </div>
          </div>
        )}
        
        <pre>{inputVideoResponse && JSON.stringify(inputVideoResponse, null, 2)}</pre>
      </div>
      
      <div className="panel grid-bottom-right">
        <h3>上傳 class 圖片</h3>
        <input type="file" accept="image/*" onChange={handleClassImgChange} />
        <button onClick={handleClassImgUpload}>上傳圖片</button>
        <pre>{classImgResponse && JSON.stringify(classImgResponse, null, 2)}</pre>
      </div>
      
      <div className="panel grid-bottom-channel-file">
        <h3>上傳 Channel Importance File</h3>
        <input type="file" onChange={handleChannelFileChange} />
        <button onClick={handleChannelFileUpload}>上傳檔案</button>
        <pre>{channelFileResponse && JSON.stringify(channelFileResponse, null, 2)}</pre>
      </div>

      <StatusPanel isStreaming={isStreaming} />
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

export default App;
