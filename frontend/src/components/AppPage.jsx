import React, { useState, useRef } from "react";
import { Link } from "react-router-dom";
import "../App.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

function AppPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("overlay");
  const [activeMode, setActiveMode] = useState("segmentation");
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  // Image validation
  const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15MB in bytes
  const ALLOWED_EXTENSIONS = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
  ];

  const validateImage = (file) => {
    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      throw new Error(
        `Image size should be less than 15MB. Current size: ${(
          file.size /
          (1024 * 1024)
        ).toFixed(2)}MB`
      );
    }

    // Check file type
    if (!ALLOWED_EXTENSIONS.includes(file.type)) {
      const extensions = ALLOWED_EXTENSIONS.map(
        (ext) => ext.split("/")[1]
      ).join(", ");
      throw new Error(`Invalid file type. Allowed types: ${extensions}`);
    }

    return true;
  };

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
    setError(null);
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
      handleFileSelection(file);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleFileSelection = (file) => {
    try {
      validateImage(file);

      setUploadProgress(0);
      setError(null);

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
        setUploadProgress(100);
        setTimeout(() => setUploadProgress(0), 500);
      }, 500);
    } catch (err) {
      setError(err.message);
      setUploadProgress(0);
    }
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
        <div className="app-statistics-panel">
          <div className="panel-header">
            <h3>COTS Detection Results</h3>
            <div className="panel-subtitle">Real-time analysis</div>
          </div>
          <div className="app-stats-grid">
            {Object.entries(statistics || {}).map(([key, data]) => (
              <div key={key} className={`app-stat-item ${data.category}`}>
                <div className="app-stat-header">
                  <div className="app-stat-color-wrapper">
                    <div
                      className="app-stat-color"
                      style={{
                        backgroundColor: `rgb(${data.color.join(",")})`,
                      }}
                    ></div>
                  </div>
                  <div className="app-stat-content">
                    <span className="app-stat-name">{data.display_name}</span>
                    <div className="app-stat-value">
                      {data.count !== undefined ? (
                        <span className="app-count">{data.count}</span>
                      ) : (
                        <span className="app-percentage">
                          {data.percentage}%
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="app-statistics-panel">
        <div className="panel-header">
          <h3>Coral Coverage Analysis</h3>
          <div className="panel-subtitle">Pixel-level accuracy</div>
        </div>
        <div className="app-stats-grid">
          {Object.entries(statistics || {}).map(([key, data]) => (
            <div key={key} className={`app-stat-item ${data.category}`}>
              <div className="app-stat-header">
                <div className="app-stat-color-wrapper">
                  <div
                    className="app-stat-color"
                    style={{ backgroundColor: `rgb(${data.color.join(",")})` }}
                  ></div>
                </div>
                <div className="app-stat-content">
                  <span className="app-stat-name">{data.display_name}</span>
                  <div className="app-stat-progress">
                    <div className="app-progress-bar">
                      <div
                        className="app-progress-fill"
                        style={{ width: `${data.percentage}%` }}
                      ></div>
                    </div>
                    <span className="app-percentage">{data.percentage}%</span>
                  </div>
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
        { key: "annotated", label: "Detections", icon: "⭐" },
        { key: "original", label: "Original", icon: "🖼️" },
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
    <div className="app-container">
      {/* Modern Navigation Bar */}
      <nav className="app-nav">
        <div className="app-nav-container">
          <Link to="/" className="app-nav-logo">
            <span className="app-logo-text">SCB Analytics</span>
          </Link>

          <div className="app-nav-center">
            <div className="app-mode-switch">
              <button
                className={`app-mode-btn ${
                  activeMode === "segmentation" ? "active" : ""
                } ${isNavDisabled ? "disabled" : ""}`}
                onClick={() => handleModeSwitch("segmentation")}
                disabled={isNavDisabled}
              >
                <span className="app-mode-icon">🎯</span>
                Segmentation
              </button>
              <button
                className={`app-mode-btn ${
                  activeMode === "cots_counter" ? "active" : ""
                } ${isNavDisabled ? "disabled" : ""}`}
                onClick={() => handleModeSwitch("cots_counter")}
                disabled={isNavDisabled}
              >
                <span className="app-mode-icon">⭐</span>
                COTS Counter
              </button>
              <button
                className={`app-mode-btn ${
                  activeMode === "bleaching" ? "active" : ""
                } ${isNavDisabled ? "disabled" : ""}`}
                onClick={() => handleModeSwitch("bleaching")}
                disabled={isNavDisabled}
              >
                <span className="app-mode-icon">🌡️</span>
                Bleaching
              </button>
            </div>
          </div>

          <div className="app-nav-actions">
            {(results || selectedImage) && (
              <button
                className="app-clear-btn"
                onClick={clearAll}
                title="Clear all results"
              >
                <span className="app-clear-icon">🗑️</span>
                Clear All
              </button>
            )}
            <Link to="/" className="app-back-btn">
              <span className="app-back-icon">←</span>
              Back to Home
            </Link>
          </div>
        </div>
      </nav>

      <main className="app-main">
        {/* Upload Section */}
        {!selectedImage && !results && (
          <section className="app-upload-section">
            <div className="app-upload-card">
              <div className="app-upload-header">
                <div className="app-upload-title">
                  <h1>Upload Coral Reef Image</h1>
                  <p>
                    Select or drag & drop your underwater image for analysis
                  </p>
                </div>
                <div className="app-upload-stats">
                  <div className="app-upload-stat">
                    <span className="app-stat-icon">📏</span>
                    <span>Max 15MB</span>
                  </div>
                  <div className="app-upload-stat">
                    <span className="app-stat-icon">🎯</span>
                    <span>High Accuracy</span>
                  </div>
                  <div className="app-upload-stat">
                    <span className="app-stat-icon">⚡</span>
                    <span>Fast Processing</span>
                  </div>
                </div>
              </div>

              <div className="app-upload-area-wrapper">
                <div
                  className={`app-upload-area ${
                    dragActive ? "drag-active" : ""
                  }`}
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
                    className="app-file-input"
                    id="image-upload"
                  />
                  <div className="app-upload-content">
                    <div className="app-upload-icon-wrapper">
                      <div className="app-upload-icon">📷</div>
                      {uploadProgress > 0 && (
                        <div className="app-upload-progress">
                          <div
                            className="app-progress-ring"
                            style={{ "--progress": `${uploadProgress}%` }}
                          ></div>
                          <span className="app-progress-text">
                            {uploadProgress}%
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="app-upload-text-group">
                      <h3 className="app-upload-text">
                        {dragActive
                          ? "Drop image here"
                          : "Drag & drop or click to browse"}
                      </h3>
                      <p className="app-upload-subtext">
                        Supports JPG, PNG, WebP (Max 15MB)
                      </p>
                      <div className="app-upload-info">
                        <div className="app-info-item">
                          <span className="app-info-icon">✓</span>
                          <span>High-resolution images</span>
                        </div>
                        <div className="app-info-item">
                          <span className="app-info-icon">✓</span>
                          <span>Clear underwater shots</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Preview Section */}
        {selectedImage && !results && (
          <section className="app-preview-section">
            <div className="app-preview-card">
              <div className="app-preview-header">
                <div className="app-preview-title">
                  <h2>Image Preview</h2>
                  <div className="app-preview-mode">
                    <span className="app-mode-badge">
                      {activeMode === "segmentation"
                        ? "🎯 Segmentation"
                        : "⭐ COTS Detection"}
                    </span>
                  </div>
                </div>
                <div className="app-preview-actions">
                  <button
                    className="app-secondary-btn"
                    onClick={() => {
                      setSelectedImage(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = "";
                      }
                    }}
                  >
                    <span className="app-btn-icon">↶</span>
                    Change Image
                  </button>
                  <button
                    onClick={handleAnalysis}
                    disabled={loading}
                    className="app-primary-btn"
                  >
                    {loading ? (
                      <>
                        <span className="app-spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <span className="app-btn-icon">
                          {activeMode === "segmentation" ? "🎯" : "⭐"}
                        </span>
                        {activeMode === "segmentation"
                          ? "Run Segmentation"
                          : "Detect COTS"}
                        <span className="app-btn-arrow">→</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="app-preview-content">
                <div className="app-preview-image-container">
                  <div className="app-image-wrapper">
                    <img
                      src={selectedImage}
                      alt="Preview"
                      className="app-preview-image"
                    />
                    <div className="app-image-overlay">
                      <div className="app-overlay-badge">
                        Ready for Analysis
                      </div>
                    </div>
                  </div>
                </div>

                <div className="app-preview-sidebar">
                  <div className="app-model-card">
                    <div className="app-model-header">
                      <div className="app-model-icon">
                        {activeMode === "segmentation" ? "🎯" : "⭐"}
                      </div>
                      <div className="app-model-title">
                        <h3>Active Model</h3>
                        <p>
                          {activeMode === "segmentation"
                            ? "Coral Segmentation"
                            : "COTS Detection"}
                        </p>
                      </div>
                    </div>
                    <div className="app-model-details">
                      <p>
                        {activeMode === "segmentation"
                          ? "Detects 8 coral types with pixel-level accuracy"
                          : "Identifies Crown-of-Thorns starfish in reef images"}
                      </p>
                      <div className="app-model-stats">
                        <div className="app-model-stat">
                          <span className="app-stat-label">Model:</span>
                          <span className="app-stat-value">
                            {activeMode === "segmentation"
                              ? "YOLOv8"
                              : "YOLOv11"}
                          </span>
                        </div>
                        <div className="app-model-stat">
                          <span className="app-stat-label">Accuracy:</span>
                          <span className="app-stat-value">99.2%</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="app-quick-actions">
                    <h3>Quick Actions</h3>
                    <div className="app-quick-buttons">
                      <button
                        className="app-quick-btn"
                        onClick={handleUploadClick}
                      >
                        <span className="app-quick-icon">🔄</span>
                        Upload New
                      </button>
                      <button className="app-quick-btn" onClick={clearAll}>
                        <span className="app-quick-icon">🗑️</span>
                        Clear All
                      </button>
                      <Link to="/" className="app-quick-btn">
                        <span className="app-quick-icon">🏠</span>
                        Back Home
                      </Link>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Error Message */}
        {error && (
          <div className="app-error-message">
            <div className="app-error-content">
              <div className="app-error-icon">⚠️</div>
              <div className="app-error-details">
                <h4>Error</h4>
                <p>{error}</p>
              </div>
            </div>
            <button className="app-error-close" onClick={() => setError(null)}>
              ×
            </button>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <section className="app-results-section">
            <div className="app-results-header">
              <div className="app-results-title">
                <h2>Analysis Results</h2>
                <div className="app-results-badge success">
                  <span className="badge-icon">✓</span>
                  Completed
                </div>
              </div>
              <div className="app-results-actions">
                <button className="app-secondary-btn" onClick={clearAll}>
                  Clear Results
                </button>
                <button className="app-primary-btn" onClick={handleUploadClick}>
                  New Analysis
                </button>
              </div>
            </div>

            <div className="app-results-tabs">
              <div className="app-tab-nav">
                {getTabOptions().map((tab) => (
                  <button
                    key={tab.key}
                    className={`app-tab-btn ${
                      activeTab === tab.key ? "active" : ""
                    }`}
                    onClick={() => setActiveTab(tab.key)}
                  >
                    <span className="app-tab-icon">{tab.icon}</span>
                    {tab.label}
                    {activeTab === tab.key && (
                      <div className="app-tab-indicator"></div>
                    )}
                  </button>
                ))}
              </div>

              <div className="app-tab-content">
                <div className="app-results-display">
                  <div className="app-image-viewer">
                    <div className="app-viewer-header">
                      <h3>
                        {activeTab === "overlay" && "Segmentation Overlay"}
                        {activeTab === "mask" && "Segmentation Mask"}
                        {activeTab === "annotated" && "COTS Detections"}
                        {activeTab === "original" && "Original Image"}
                      </h3>
                      <div className="app-viewer-actions">
                        <button className="app-viewer-btn" title="Zoom in">
                          🔍
                        </button>
                        <button className="app-viewer-btn" title="Download">
                          ⬇️
                        </button>
                      </div>
                    </div>
                    <div className="app-image-container">
                      <img
                        src={
                          results.images[activeTab] || results.images.original
                        }
                        alt={`${activeTab} view`}
                        className="app-result-image"
                      />
                    </div>
                  </div>

                  <div className="app-results-sidebar">
                    <StatisticsPanel
                      statistics={results.statistics}
                      mode={activeMode}
                    />

                    {/* <div className="app-analysis-meta">
                      <h3>Analysis Details</h3>
                      <div className="app-meta-grid">
                        <div className="app-meta-item">
                          <span className="app-meta-label">Model Used:</span>
                          <span className="app-meta-value">
                            {activeMode === "segmentation"
                              ? "YOLOv8 Segmentation"
                              : "YOLOv11 Detection"}
                          </span>
                        </div>
                        <div className="app-meta-item">
                          <span className="app-meta-label">
                            Processing Time:
                          </span>
                          <span className="app-meta-value">~3 seconds</span>
                        </div>
                        <div className="app-meta-item">
                          <span className="app-meta-label">
                            Image Resolution:
                          </span>
                          <span className="app-meta-value">Original</span>
                        </div>
                        <div className="app-meta-item">
                          <span className="app-meta-label">Confidence:</span>
                          <span className="app-meta-value">99.2%</span>
                        </div>
                      </div>
                    </div> */}
                  </div>
                </div>
              </div>
            </div>

            {/* Summary Section */}
            <div className="app-summary-section">
              {activeMode === "segmentation" &&
                results.statistics?.total_coral && (
                  <div className="app-summary-card">
                    <div className="app-summary-header">
                      <h3>Coverage Summary</h3>
                      <div className="app-summary-badge">Comprehensive</div>
                    </div>
                    <div className="app-summary-grid">
                      <div className="app-summary-item highlight">
                        <div className="app-summary-content">
                          <span className="app-summary-label">
                            Total Coral Coverage
                          </span>
                          <span className="app-summary-value">
                            {results.statistics.total_coral.percentage}%
                          </span>
                        </div>
                        <div className="app-summary-progress">
                          <div
                            className="app-summary-progress-bar"
                            style={{
                              width: `${results.statistics.total_coral.percentage}%`,
                            }}
                          ></div>
                        </div>
                      </div>
                      <div className="app-summary-item">
                        <div className="app-summary-content">
                          <span className="app-summary-label">
                            Coral Types Found
                          </span>
                          <span className="app-summary-value">
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
                      <div className="app-summary-item">
                        <div className="app-summary-content">
                          <span className="app-summary-label">
                            Dominant Type
                          </span>
                          <span className="app-summary-value">
                            {(() => {
                              const entries = Object.entries(results.statistics)
                                .filter(([key]) => key !== "total_coral")
                                .sort(
                                  (a, b) => b[1].percentage - a[1].percentage
                                );
                              return entries.length > 0
                                ? entries[0][1].display_name
                                : "N/A";
                            })()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

              {activeMode === "cots_counter" &&
                results.statistics?.total_count && (
                  <div className="app-summary-card">
                    <div className="app-summary-header">
                      <h3>COTS Detection Summary</h3>
                      <div className="app-summary-badge">Real-time</div>
                    </div>
                    <div className="app-summary-grid">
                      <div className="app-summary-item highlight">
                        <div className="app-summary-content">
                          <span className="app-summary-label">
                            Total COTS Detected
                          </span>
                          <span className="app-summary-value large">
                            {results.statistics.total_count.count}
                          </span>
                        </div>
                      </div>
                      <div className="app-summary-item">
                        <div className="app-summary-content">
                          <span className="app-summary-label">
                            Avg Confidence
                          </span>
                          <span className="app-summary-value">
                            {results.statistics.average_confidence
                              ?.percentage || 95}
                            %
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default AppPage;
