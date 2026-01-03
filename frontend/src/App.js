// frontend/src/App.js
import React, { useState, useRef } from "react";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("overlay");
  const [activeMode, setActiveMode] = useState("segmentation");
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const clearAll = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
    setUploadProgress(0);
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
        handleFileSelection(file);
      } else {
        setError("Please upload an image file");
      }
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleFileSelection = (file) => {
    // Simulate upload progress
    setUploadProgress(0);
    const timer = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          return 100;
        }
        return prev + 10;
      });
    }, 50);

    setTimeout(() => {
      setSelectedImage(URL.createObjectURL(file));
      setResults(null);
      setError(null);
      setUploadProgress(100);
      setTimeout(() => setUploadProgress(0), 500);
    }, 500);
  };

  const handleUploadClick = () => {
    if (isNavDisabled) return;
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
          <div className="panel-header">
            <h3>COTS Detection Results</h3>
          </div>
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
        <div className="panel-header">
          <h3>Coral Coverage Analysis</h3>
          <div className="panel-badge">Detailed</div>
        </div>
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
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${data.percentage}%` }}
                  ></div>
                </div>
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
        { key: "annotated", label: "Detections" },
        { key: "original", label: "Original" },
      ];
    }
    return [
      { key: "overlay", label: "Overlay" },
      { key: "mask", label: "Mask" },
      { key: "original", label: "Original" },
    ];
  };

  const isNavDisabled = results || loading;

  return (
    <div className="App">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="navbar-container">
          <div className="navbar-brand">
            <div className="logo">
              <div>
                <h1>DeepCoralAI V1.1</h1>
              </div>
            </div>
          </div>

          <div className="navbar-menu">
            <div className="nav-mode-switch">
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
                Bleaching
              </button>
            </div>

            {(results || selectedImage) && (
              <button
                className="clear-btn-nav"
                onClick={clearAll}
                title="Clear all results"
              >
                <span className="clear-icon">🗑️</span>
                Clear All
              </button>
            )}
          </div>
        </div>
      </nav>

      <div className="main-container">
        {/* Upload Section - Only shown when no image is selected */}
        {!selectedImage && !results && (
          <div className="upload-section">
            <div className="upload-card">
              <div className="upload-header">
                <h2>Upload Underwater Coral Image</h2>
              </div>

              <div className="upload-area-wrapper">
                <div
                  className={`upload-area ${dragActive ? "drag-active" : ""}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={handleUploadClick}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="file-input"
                    id="image-upload"
                  />
                  <div className="upload-content">
                    <div className="upload-icon-wrapper">
                      <div className="upload-icon">📷</div>
                      {uploadProgress > 0 && (
                        <div className="upload-progress">
                          <div
                            className="progress-circle"
                            style={{ "--progress": `${uploadProgress}%` }}
                          ></div>
                        </div>
                      )}
                    </div>
                    <div className="upload-text-group">
                      <p className="upload-text">
                        {dragActive
                          ? "Drop image here"
                          : "Drag & drop or click to browse"}
                      </p>
                      <p className="upload-subtext">
                        Supports JPG, PNG, WebP up to 10MB
                      </p>
                    </div>
                  </div>
                </div>
                {/* <div className="upload-tips">
                  <div className="tip">
                    Use high-resolution images for best results
                  </div>
                  <div className="tip">Processing time: 3-10 seconds</div>
                </div> */}
              </div>
            </div>
          </div>
        )}

        {selectedImage && (
          <div className="preview-section">
            <div className="preview-container">
              <div className="preview-header">
                <div className="preview-title">
                  <h2>Image Preview</h2>
                  <div className="preview-meta">
                    <span className="meta-item">
                      <span className="meta-icon">📏</span>
                      {activeMode === "segmentation"
                        ? "Segmentation Mode"
                        : "COTS Detection Mode"}
                    </span>
                    <span className="meta-item">
                      <span className="meta-icon">⏱</span>
                      Ready for Analysis
                    </span>
                  </div>
                </div>
                <div className="preview-actions">
                  <button
                    className="action-btn secondary"
                    onClick={() => {
                      setSelectedImage(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = "";
                      }
                    }}
                    title="Remove this image"
                  >
                    <span className="action-icon">↶</span>
                    Change Image
                  </button>
                  <button
                    onClick={handleAnalysis}
                    disabled={loading}
                    className="action-btn primary"
                  >
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        {/* <span className="action-icon">
                          {activeMode === "segmentation" ? "🎯" : "⭐"}
                        </span> */}
                        {activeMode === "segmentation"
                          ? "Run Segmentation"
                          : "Detect COTS"}
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="preview-content">
                <div className="preview-image-container">
                  <img
                    src={selectedImage}
                    alt="Preview"
                    className="preview-image"
                  />
                  {/* <div className="preview-overlay">
                    <div className="preview-info">
                      <div className="info-badge">Selected</div>
                      <div className="info-text">Ready for processing</div>
                    </div>
                  </div> */}
                </div>

                <div className="preview-sidebar">
                  <div className="model-info">
                    <h3>Active Model</h3>
                    <div className="model-card">
                      {/* <div className="model-icon">
                        {activeMode === "segmentation" ? "🎯" : "⭐"}
                      </div> */}
                      <div className="model-details">
                        <h4>
                          {activeMode === "segmentation"
                            ? "Coral Segmentation"
                            : "COTS Detection"}
                        </h4>
                        <p className="model-desc">
                          {activeMode === "segmentation"
                            ? "Detects 8 coral types with pixel-level accuracy"
                            : "Identifies Crown-of-Thorns starfish in reef images"}
                        </p>
                        <div className="model-stats">
                          <span className="model-stat">
                            <strong>Type:</strong>{" "}
                            {activeMode === "segmentation"
                              ? "YOLOv8"
                              : "YOLOv11"}
                          </span>
                          {/* <span className="model-stat">
                            <strong>Precision:</strong> 95%
                          </span> */}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="quick-actions">
                    <h3>Quick Actions</h3>
                    <div className="action-buttons-grid">
                      <button
                        className="quick-action-btn"
                        onClick={handleUploadClick}
                      >
                        <span className="quick-icon">🔄</span>
                        Upload New
                      </button>
                      <button className="quick-action-btn" onClick={clearAll}>
                        <span className="quick-icon">🗑️</span>
                        Clear All
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            <div className="error-content">
              <span className="error-icon">⚠️</span>
              <div className="error-details">
                <h4>Analysis Failed</h4>
                <p>{error}</p>
              </div>
            </div>
            <button className="clear-error-btn" onClick={() => setError(null)}>
              ×
            </button>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div className="results-section">
            <div className="results-header">
              <div className="results-title">
                <div className="title-group">
                  <h2>Analysis Results</h2>
                  <div className="result-badge success">Completed</div>
                </div>
                {/* <div className="result-meta">
                  <span className="meta-item">
                    <span className="meta-icon">📅</span>
                    Just now
                  </span>
                </div> */}
              </div>

              <div className="results-actions">
                <button className="action-btn secondary" onClick={clearAll}>
                  <span className="action-icon">🗑️</span>
                  Clear Results
                </button>
                {/* <button
                  className="action-btn primary"
                  onClick={handleUploadClick}
                >
                  <span className="action-icon">📷</span>
                  New Analysis
                </button> */}
              </div>
            </div>

            <div className="tabs-container">
              <div className="tab-navigation">
                {getTabOptions().map((tab) => (
                  <button
                    key={tab.key}
                    className={`tab-btn ${
                      activeTab === tab.key ? "active" : ""
                    }`}
                    onClick={() => setActiveTab(tab.key)}
                  >
                    <span className="tab-icon">{tab.icon}</span>
                    {tab.label}
                    {activeTab === tab.key && (
                      <div className="tab-indicator"></div>
                    )}
                  </button>
                ))}
              </div>

              <div className="tab-content">
                <div className="results-display">
                  <div className="image-viewer">
                    <div className="viewer-header">
                      <h3>
                        {activeTab === "overlay" && "Segmentation Overlay"}
                        {activeTab === "mask" && "Segmentation Mask"}
                        {activeTab === "annotated" && "COTS Detections"}
                        {activeTab === "original" && "Original Image"}
                      </h3>
                      {/* <div className="viewer-actions">
                        <button className="viewer-btn" title="Zoom in">
                          🔍
                        </button>
                        <button className="viewer-btn" title="Download">
                          ⬇️
                        </button>
                      </div> */}
                    </div>
                    <div className="image-container">
                      <img
                        src={
                          results.images[activeTab] || results.images.original
                        }
                        alt={`${activeTab} view`}
                        className="result-image"
                      />
                    </div>
                  </div>

                  <div className="results-sidebar">
                    <StatisticsPanel
                      statistics={results.statistics}
                      mode={activeMode}
                    />

                    <div className="analysis-meta">
                      <h3>Analysis Details</h3>
                      <div className="meta-grid">
                        <div className="meta-item">
                          <span className="meta-label">Model Used:</span>
                          <span className="meta-value">
                            {activeMode === "segmentation"
                              ? "YOLOv8 Segmentation"
                              : "YOLOv11 Detection"}
                          </span>
                        </div>
                        <div className="meta-item">
                          <span className="meta-label">Processing Time:</span>
                          <span className="meta-value">~3 seconds</span>
                        </div>
                        <div className="meta-item">
                          <span className="meta-label">Image Size:</span>
                          <span className="meta-value">Original</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Summary Card */}
            <div className="summary-section">
              {activeMode === "segmentation" &&
                results.statistics &&
                results.statistics.total_coral && (
                  <div className="summary-card">
                    <div className="summary-header">
                      <h3>Coverage Summary</h3>
                      <div className="summary-badge">Comprehensive</div>
                    </div>
                    <div className="summary-grid">
                      <div className="summary-item highlight">
                        <div className="summary-content">
                          <span className="summary-label">
                            Total Coral Coverage
                          </span>
                          <span className="summary-value">
                            {results.statistics.total_coral.percentage}%
                          </span>
                        </div>
                      </div>
                      <div className="summary-item">
                        <div className="summary-content">
                          <span className="summary-label">
                            Coral Types Found
                          </span>
                          <span className="summary-value">
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
                      {/* <div className="summary-item">
                        <div className="summary-icon">📈</div>
                        <div className="summary-content">
                          <span className="summary-label">Dominant Type</span>
                          <span className="summary-value">
                            {
                              Object.entries(results.statistics)
                                .filter(([key]) => key !== "total_coral")
                                .reduce((max, curr) =>
                                  curr[1].percentage > max[1].percentage
                                    ? curr
                                    : max
                                )[1].display_name
                            }
                          </span>
                        </div>
                      </div> */}
                    </div>
                  </div>
                )}

              {activeMode === "cots_counter" &&
                results.statistics &&
                results.statistics.total_count && (
                  <div className="summary-card">
                    <div className="summary-header">
                      <h3>COTS Detection Summary</h3>
                    </div>
                    <div className="summary-grid">
                      <div className="summary-item highlight">
                        <div className="summary-icon">⚠️</div>
                        <div className="summary-content">
                          <span className="summary-label">
                            Total COTS Detected
                          </span>
                          <span className="summary-value large">
                            {results.statistics.total_count.count}
                          </span>
                        </div>
                      </div>
                      <div className="summary-item">
                        <div className="summary-icon">🎯</div>
                        <div className="summary-content">
                          <span className="summary-label">Avg Confidence</span>
                          <span className="summary-value">
                            {results.statistics.average_confidence.percentage}%
                          </span>
                        </div>
                      </div>
                      {/* <div className="summary-item">
                        <div className="summary-icon">📊</div>
                        <div className="summary-content">
                          <span className="summary-label">Risk Level</span>
                          <span
                            className={`summary-value ${
                              results.statistics.total_count.count > 10
                                ? "danger"
                                : results.statistics.total_count.count > 5
                                ? "warning"
                                : "safe"
                            }`}
                          >
                            {results.statistics.total_count.count > 10
                              ? "High"
                              : results.statistics.total_count.count > 5
                              ? "Medium"
                              : "Low"}
                          </span>
                        </div>
                      </div> */}
                    </div>
                  </div>
                )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
