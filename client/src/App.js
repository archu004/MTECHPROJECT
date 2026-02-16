import React, { useRef, useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [age, setAge] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // 🔹 Open webcam
  const openCamera = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;

      if (!videoRef.current) throw new Error("Video element not ready");

      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraOn(true);
    } catch (err) {
      console.error("Camera error:", err);
      setError(
        err.name === "NotAllowedError"
          ? "Camera access denied by user"
          : "Camera not supported or in use"
      );
    }
  };

  // 🔹 Capture photo
  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      setImage(file);
      closeCamera();
    }, "image/jpeg");
  };

  // 🔹 Stop webcam
  const closeCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    setCameraOn(false);
  };

  // 🔹 Predict age
  const predictAge = async () => {
    setError("");
    setResult(null);

    if (!image || !age) {
      setError("Please upload or capture an image and enter age");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);
    formData.append("chronological_age", age);

    try {
      setLoading(true);
      const response = await fetch("http://127.0.0.1:8000/predict-age", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Prediction failed");

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h2 className="title">🧬 Biological Age Prediction</h2>

        {/* File Upload & Webcam Buttons */}
        <div className="button-row">
          <input
            type="file"
            accept="image/*"
            id="file-upload"
            style={{ display: "none" }}
            onChange={(e) => setImage(e.target.files[0])}
          />

          <label htmlFor="file-upload" className="camera-button">
            🖼️ Upload Image
          </label>

          <button className="secondary-button" onClick={openCamera}>
            📹 Use Webcam
          </button>
        </div>

        {/* Webcam Preview */}
        <div
          className="camera-box"
          style={{ display: cameraOn ? "block" : "none" }}
        >
          <video ref={videoRef} autoPlay playsInline className="video" />
          <div className="button-row">
            <button className="button" onClick={capturePhoto}>
              Capture
            </button>
            <button className="secondary-button" onClick={closeCamera}>
              Cancel
            </button>
          </div>
        </div>

        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Image Preview */}
        {image && (
          <img
            src={URL.createObjectURL(image)}
            alt="preview"
            className="preview"
          />
        )}

        {/* Age Input */}
        <input
          type="number"
          placeholder="Enter chronological age"
          value={age}
          className="input"
          onChange={(e) => setAge(e.target.value)}
        />

        {/* Predict Button */}
        <button className="button" onClick={predictAge} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>

        {/* Error */}
        {error && <p className="error">{error}</p>}

        {/* ========================= */}
        {/* Prediction Result */}
        {/* ========================= */}

        {result && (
  <div className="result-wrapper">
    
    {/* LEFT SIDE - Prediction Summary */}
    <div className="result-card">
      <h3>📊 Result</h3>

      <p>
        <b>Predicted Group:</b> {result.predicted_group}
      </p>

      <p>
        <b>Biological Age:</b> {result.biological_age}
      </p>

      <p>
        <b>Deviation:</b>{" "}
        <span className={result.deviation_years < 0 ? "good" : "bad"}>
          {result.deviation_years} years
        </span>
      </p>
    </div>

    {/* RIGHT SIDE - Health Analysis */}
    {result.health_analysis && (
      <div className="health-card">
        <h3>🩺 Health Analysis</h3>

        <p>
          <b>Status:</b> {result.health_analysis.status}
        </p>

        <p>
          <b>Risk Level:</b>{" "}
          <span
            className={
              result.health_analysis.risk_level === "Elevated"
                ? "bad"
                : result.health_analysis.risk_level === "Good"
                ? "good"
                : ""
            }
          >
            {result.health_analysis.risk_level}
          </span>
        </p>

        <div className="suggestion-block">
          <h5>🥗 Food Suggestions</h5>
          <ul>
            {result.health_analysis.food_suggestions.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>

        <div className="suggestion-block">
          <h5>🏃 Lifestyle Suggestions</h5>
          <ul>
            {result.health_analysis.lifestyle_suggestions.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
    )}
  </div>
)}

      </div>
    </div>
  );
}

export default App;
