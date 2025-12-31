// frontend/src/App.js
import React, { useState, useRef } from "react";
import "./App.css";

const API_BASE = "http://localhost:5000/api";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("overlay");
  const [activeMode, setActiveMode] = useState("segmentation");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const clearAll = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
    setActiveTab("overlay");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleModeSwitch = (mode) => {
    if (results || loading) {
      return;
    }
    setActiveMode(mode);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        setSelectedImage(URL.createObjectURL(file));
        setResults(null);
        setError(null);
      } else {
        setError("Please upload an image file");
      }
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleAnalysis = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      formData.append("image", blob, "image.png");

      let endpoint = `${API_BASE}/segment`;
      if (activeMode === "cots_counter") {
        endpoint = `${API_BASE}/cots-counter`;
      }

      const result = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!result.ok) {
        const errorData = await result.json();
        throw new Error(errorData.error || "Analysis failed");
      }

      const data = await result.json();
      setResults(data);

      if (activeMode === "cots_counter") {
        setActiveTab("annotated");
      } else {
        setActiveTab("overlay");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const StatisticsPanel = ({ statistics, mode }) => {
    if (mode === "cots_counter") {
      return (
        <div className="statistics-panel">
          <h3>COTS Detection Results</h3>
          <div className="stats-grid">
            {Object.entries(statistics || {}).map(([key, data]) => (
              <div key={key} className={`stat-item ${data.category}`}>
                <div className="stat-header">
                  <div
                    className="stat-color"
                    style={{ backgroundColor: `rgb(${data.color.join(",")})` }}
                  ></div>
                  <span className="stat-name">{data.display_name}</span>
                </div>
                <div className="stat-values">
                  {data.count !== undefined ? (
                    <span className="count">{data.count}</span>
                  ) : (
                    <span className="percentage">{data.percentage}%</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="statistics-panel">
        <h3>Coral Coverage Analysis</h3>
        <div className="stats-grid">
          {Object.entries(statistics || {}).map(([key, data]) => (
            <div key={key} className={`stat-item ${data.category}`}>
              <div className="stat-header">
                <div
                  className="stat-color"
                  style={{ backgroundColor: `rgb(${data.color.join(",")})` }}
                ></div>
                <span className="stat-name">{data.display_name}</span>
              </div>
              <div className="stat-values">
                <span className="percentage">{data.percentage}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const getTabOptions = () => {
    if (activeMode === "cots_counter") {
      return [
        { key: "annotated", label: "Detections", icon: "🔍" },
        { key: "original", label: "Original", icon: "📷" },
      ];
    }
    return [
      { key: "overlay", label: "Overlay", icon: "🎨" },
      { key: "mask", label: "Mask", icon: "🎭" },
      { key: "original", label: "Original", icon: "📷" },
    ];
  };

  const isNavDisabled = results || loading;

  return (
    <div className="App">
      <nav className="navbar">
        <div className="navbar-container">
          <div className="navbar-brand">
            <h1>DeepCoralAI V2</h1>
          </div>

          <div className="navbar-menu">
            <button
              className={`nav-btn ${
                activeMode === "segmentation" ? "active" : ""
              } ${isNavDisabled ? "disabled" : ""}`}
              onClick={() => handleModeSwitch("segmentation")}
              disabled={isNavDisabled}
              title={isNavDisabled ? "Clear results to switch mode" : ""}
            >
              Segmentation
            </button>
            <button
              className={`nav-btn ${
                activeMode === "cots_counter" ? "active" : ""
              } ${isNavDisabled ? "disabled" : ""}`}
              onClick={() => handleModeSwitch("cots_counter")}
              disabled={isNavDisabled}
              title={isNavDisabled ? "Clear results to switch mode" : ""}
            >
              COTS Counter
            </button>
            <button
              className={`nav-btn ${
                activeMode === "bleaching" ? "active" : ""
              } ${isNavDisabled ? "disabled" : ""}`}
              onClick={() => handleModeSwitch("bleaching")}
              disabled={isNavDisabled}
              title={isNavDisabled ? "Clear results to switch mode" : ""}
            >
              Coral Bleaching
            </button>

            {/* Clear Button in Navbar */}
            {(results || selectedImage) && (
              <button
                className="clear-btn-nav"
                onClick={clearAll}
                title="Clear all results"
              >
                🗑️ Clear
              </button>
            )}
          </div>
        </div>
      </nav>

      <div className="main-container">
        <div className="mode-indicator">
          <div className={`mode-tag ${activeMode}`}>
            {activeMode === "segmentation" && "🎯 Coral Segmentation"}
            {activeMode === "cots_counter" &&
              "⭐ Crown-of-Thorns Starfish Detection"}
            {activeMode === "bleaching" && "🌡️ Coral Bleaching Analysis"}
          </div>
          <p className="mode-description">
            {activeMode === "segmentation" &&
              "Upload coral reef images for automatic segmentation and coverage analysis"}
            {activeMode === "cots_counter" &&
              "Detect and count Crown-of-Thorns starfish in reef images"}
            {activeMode === "bleaching" &&
              "Analyze coral health and detect bleaching patterns"}
          </p>
        </div>

        <div className="upload-section">
          <div className="upload-card">
            <div className="upload-header">
              <h2>Upload Underwater Image</h2>
              {selectedImage && (
                <button
                  className="clear-btn-upload"
                  onClick={clearAll}
                  title="Clear image and results"
                >
                  Clear All
                </button>
              )}
            </div>
            <p>Drag & drop or click to upload a coral reef image</p>

            <div
              className={`upload-area ${dragActive ? "drag-active" : ""} ${
                isNavDisabled ? "upload-disabled" : ""
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={isNavDisabled ? null : handleUploadClick}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
                id="image-upload"
                disabled={isNavDisabled}
              />
              <div className="upload-content">
                <div className="upload-icon">📷</div>
                <p className="upload-text">
                  {dragActive
                    ? "Drop image here"
                    : isNavDisabled
                    ? "Clear current results to upload new image"
                    : "Drag & drop or click to browse"}
                </p>
                <p className="upload-subtext">
                  Supports JPG, PNG, WebP up to 10MB
                </p>
              </div>
            </div>

            {selectedImage && (
              <>
                <div className="image-preview">
                  <div className="preview-header">
                    <h4>Selected Image</h4>
                    <button
                      className="clear-preview-btn"
                      onClick={() => {
                        setSelectedImage(null);
                        if (fileInputRef.current) {
                          fileInputRef.current.value = "";
                        }
                      }}
                      title="Remove this image"
                    >
                      ×
                    </button>
                  </div>
                  <div className="preview-container">
                    <img
                      src={selectedImage}
                      alt="Preview"
                      className="preview-img"
                    />
                  </div>
                </div>

                <div className="action-buttons">
                  <button
                    onClick={handleAnalysis}
                    disabled={loading}
                    className="analyze-btn"
                  >
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        {activeMode === "segmentation" && "🎯 Segment Coral"}
                        {activeMode === "cots_counter" && "⭐ Count COTS"}
                        {activeMode === "bleaching" && "🌡️ Analyze Bleaching"}
                      </>
                    )}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="error-message">
            <span className="error-icon">❌</span>
            Error: {error}
            <button className="clear-error-btn" onClick={() => setError(null)}>
              ×
            </button>
          </div>
        )}

        {results && (
          <div className="results-section">
            <div className="results-header">
              <div className="results-title">
                <h2>Analysis Results</h2>
                <button
                  className="clear-results-header-btn"
                  onClick={clearAll}
                  title="Clear all results"
                >
                  Clear All
                </button>
              </div>
              <div className="tab-navigation">
                {getTabOptions().map((tab) => (
                  <button
                    key={tab.key}
                    className={activeTab === tab.key ? "active" : ""}
                    onClick={() => setActiveTab(tab.key)}
                  >
                    {tab.icon && <span className="tab-icon">{tab.icon}</span>}
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="results-content">
              <div className="image-display">
                <div className="result-image-container">
                  <img
                    src={results.images[activeTab] || results.images.original}
                    alt={`${activeTab} view`}
                    className="result-image"
                  />
                </div>
              </div>

              <div className="side-panels">
                <StatisticsPanel
                  statistics={results.statistics}
                  mode={activeMode}
                />
              </div>
            </div>

            {activeMode === "segmentation" &&
              results.statistics &&
              results.statistics.total_coral && (
                <div className="summary-card">
                  <div className="summary-content">
                    <div className="summary-header">
                      <h3>📊 Quick Summary</h3>
                      <button
                        className="clear-summary-btn"
                        onClick={clearAll}
                        title="Clear all results"
                      >
                        Clear All
                      </button>
                    </div>
                    <div className="summary-stats">
                      <div className="total-coverage">
                        <span className="label">Total Coral Coverage:</span>
                        <span className="value">
                          {results.statistics.total_coral.percentage}%
                        </span>
                      </div>
                      <div className="coral-types">
                        <span className="label">Coral Types Detected:</span>
                        <span className="value">
                          {
                            Object.keys(results.statistics).filter(
                              (key) =>
                                key !== "total_coral" &&
                                results.statistics[key].percentage > 0
                            ).length
                          }
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

            {activeMode === "cots_counter" &&
              results.statistics &&
              results.statistics.total_count && (
                <div className="summary-card">
                  <div className="summary-content">
                    <div className="summary-header">
                      <h3>⭐ COTS Detection Summary</h3>
                      <button
                        className="clear-summary-btn"
                        onClick={clearAll}
                        title="Clear all results"
                      >
                        Clear All
                      </button>
                    </div>
                    <div className="summary-stats">
                      <div className="total-count">
                        <span className="label">Total COTS Detected:</span>
                        <span className="value">
                          {results.statistics.total_count.count}
                        </span>
                      </div>
                      <div className="avg-confidence">
                        <span className="label">Average Confidence:</span>
                        <span className="value">
                          {results.statistics.average_confidence.percentage}%
                        </span>
                      </div>
                    </div>
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
