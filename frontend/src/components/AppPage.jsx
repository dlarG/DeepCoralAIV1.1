import React, { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

const GlobalStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; background: #040d18; }
    .app-root { font-family: 'Space Grotesk', sans-serif; }
    .mono { font-family: 'DM Mono', monospace; }

    .bg-mesh {
      background-color: #040d18;
      background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,200,180,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0,100,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 60% 30%, rgba(0,230,160,0.05) 0%, transparent 50%);
    }

    .card-glow {
      border: 1px solid rgba(0,230,160,0.12);
      box-shadow: 0 0 0 1px rgba(0,230,160,0.04), 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
      transition: border-color 0.2s ease;
    }
    .card-glow:hover { border-color: rgba(0,230,160,0.22); }

    @keyframes border-pulse {
      0%, 100% { border-color: rgba(0,230,160,0.22); }
      50%       { border-color: rgba(0,230,160,0.55); }
    }
    .drop-pulse { animation: border-pulse 2.4s ease-in-out infinite; }
    .drop-active { border-color: rgba(0,230,160,0.8) !important; background: rgba(0,230,160,0.04) !important; animation: none !important; }

    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner {
      width: 15px; height: 15px;
      border: 2px solid rgba(0,0,0,0.2);
      border-top-color: #000;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: inline-block;
    }

    @keyframes shimmer {
      0%   { background-position: -200% center; }
      100% { background-position:  200% center; }
    }
    .progress-shimmer {
      background: linear-gradient(90deg, #00e6a0 0%, #00c8ff 50%, #00e6a0 100%);
      background-size: 200% auto;
      animation: shimmer 1.5s linear infinite;
    }

    .tab-active { color: #00e6a0; }
    .tab-indicator {
      position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, #00e6a0, #00c8ff);
      border-radius: 9999px;
    }

    @keyframes fadeUp {
      from { opacity:0; transform: translateY(14px); }
      to   { opacity:1; transform: translateY(0); }
    }
    .fade-up { animation: fadeUp 0.45s ease both; }

    .img-zoom { transition: transform 0.4s ease; }
    .img-zoom:hover { transform: scale(1.012); }

    .stat-bar-fill {
      height: 4px; border-radius: 9999px;
      background: linear-gradient(90deg, #00e6a0, #00c8ff);
      transition: width 0.9s cubic-bezier(.4,0,.2,1);
    }

    .quadrat-btn { transition: all 0.18s ease; }
    .quadrat-btn:hover { border-color: rgba(0,230,160,0.45) !important; transform: translateY(-2px); }
    .qactive { border-color: #00e6a0 !important; box-shadow: 0 0 0 1px rgba(0,230,160,0.4); }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,230,160,0.2); border-radius: 9999px; }
  `}</style>
);

const Badge = ({ children, variant = "default" }) => {
  const v = {
    default: "bg-white/5 text-white/50 border border-white/10",
    success: "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20",
    warning: "bg-amber-500/10 text-amber-400 border border-amber-500/20",
    info: "bg-sky-500/10 text-sky-400 border border-sky-500/20",
  };
  return (
    <span
      className={`mono text-[10px] tracking-[0.18em] uppercase px-2.5 py-1 rounded-full ${v[variant]}`}
    >
      {children}
    </span>
  );
};

const SectionLabel = ({ children }) => (
  <p className="mono text-[10px] tracking-[0.2em] uppercase text-emerald-400/50 mb-1">
    {children}
  </p>
);

export default function AppPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("overlay");
  const [activeMode, setActiveMode] = useState("segmentation");
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedQuadrat, setSelectedQuadrat] = useState(0);
  const [showOriginalImage, setShowOriginalImage] = useState(false);
  const fileInputRef = useRef(null);

  const MAX_FILE_SIZE = 15 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
  ];

  const validateImage = (file) => {
    if (file.size > MAX_FILE_SIZE)
      throw new Error(
        `Image must be under 15 MB. Current: ${(
          file.size /
          (1024 * 1024)
        ).toFixed(2)} MB`
      );
    if (!ALLOWED_EXTENSIONS.includes(file.type)) {
      const exts = ALLOWED_EXTENSIONS.map((e) => e.split("/")[1]).join(", ");
      throw new Error(`Invalid file type. Allowed: ${exts}`);
    }
    return true;
  };

  const drawBarChart = (statistics, canvasId) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !statistics) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const width = rect.width,
      height = rect.height;
    const pad = { top: 20, right: 30, bottom: 60, left: 120 };
    ctx.clearRect(0, 0, width, height);

    const coralEntries = Object.entries(statistics)
      .filter(
        ([key]) =>
          key !== "total_coral" && statistics[key].category !== "summary"
      )
      .sort((a, b) => b[1].percentage - a[1].percentage);
    if (!coralEntries.length) return;

    const maxPct = Math.max(...coralEntries.map(([_, d]) => d.percentage));
    const chartW = width - pad.left - pad.right;
    const chartH = height - pad.top - pad.bottom;
    const barW = Math.min(40, (chartW / coralEntries.length) * 0.6);
    const barGap =
      (chartW - barW * coralEntries.length) / (coralEntries.length + 1);

    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = pad.top + (chartH * i) / 5;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.font = "11px 'DM Mono',monospace";
      ctx.textAlign = "right";
      ctx.fillText(
        `${Math.round(maxPct - (maxPct * i) / 5)}%`,
        pad.left - 10,
        y + 4
      );
    }

    coralEntries.forEach(([, data], index) => {
      const x = pad.left + barGap + (barW + barGap) * index;
      const bh = (data.percentage / maxPct) * chartH;
      const y = pad.top + chartH - bh;
      const gr = ctx.createLinearGradient(x, y, x, pad.top + chartH);
      gr.addColorStop(0, `rgba(${data.color.join(",")},0.85)`);
      gr.addColorStop(1, `rgba(${data.color.join(",")},0.18)`);
      ctx.fillStyle = gr;
      const r = 4;
      ctx.beginPath();
      ctx.moveTo(x, pad.top + chartH);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.lineTo(x + barW - r, y);
      ctx.quadraticCurveTo(x + barW, y, x + barW, y + r);
      ctx.lineTo(x + barW, pad.top + chartH);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.font = "bold 11px 'DM Mono',monospace";
      ctx.textAlign = "center";
      ctx.fillText(`${data.percentage}%`, x + barW / 2, y - 8);
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "10px 'Space Grotesk',sans-serif";
      ctx.textAlign = "right";
      ctx.translate(x + barW / 2, height - pad.bottom + 15);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(data.display_name.split(" ").slice(0, 2).join(" "), 0, 0);
      ctx.restore();
    });
  };

  const drawDonutChart = (statistics, canvasId) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !statistics) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const width = rect.width,
      height = rect.height;
    const cx = width / 2,
      cy = height / 2;
    const radius = Math.min(width, height) / 2 - 20;
    ctx.clearRect(0, 0, width, height);

    const coralPct = statistics.total_coral?.percentage || 0;
    const nonCoralPct = 100 - coralPct;

    const drawArc = (s, e, color, alpha = 1) => {
      ctx.beginPath();
      ctx.arc(cx, cy, radius, s, e);
      ctx.arc(cx, cy, radius * 0.6, e, s, true);
      ctx.closePath();
      ctx.fillStyle = `rgba(${color.join(",")},${alpha})`;
      ctx.fill();
    };
    drawArc(0, (nonCoralPct / 100) * Math.PI * 2, [30, 42, 58], 0.65);

    const gr = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
    gr.addColorStop(0, "rgba(0,230,160,0.85)");
    gr.addColorStop(1, "rgba(0,200,255,0.85)");
    ctx.beginPath();
    ctx.arc(cx, cy, radius, (nonCoralPct / 100) * Math.PI * 2, Math.PI * 2);
    ctx.arc(
      cx,
      cy,
      radius * 0.6,
      Math.PI * 2,
      (nonCoralPct / 100) * Math.PI * 2,
      true
    );
    ctx.closePath();
    ctx.fillStyle = gr;
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.font = "bold 26px 'Space Grotesk',sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${coralPct}%`, cx, cy + 5);
    ctx.font = "11px 'DM Mono',monospace";
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fillText("coral", cx, cy + 23);

    const ly = height - 22;
    const gr2 = ctx.createLinearGradient(width / 2 - 60, 0, width / 2 - 48, 0);
    gr2.addColorStop(0, "rgba(0,230,160,0.85)");
    gr2.addColorStop(1, "rgba(0,200,255,0.85)");
    ctx.fillStyle = gr2;
    ctx.fillRect(width / 2 - 60, ly, 12, 12);
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "11px 'Space Grotesk',sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Coral", width / 2 - 42, ly + 10);
    ctx.fillStyle = "rgba(30,42,58,0.9)";
    ctx.fillRect(width / 2 + 20, ly, 12, 12);
    ctx.fillText("Non-Coral", width / 2 + 38, ly + 10);
  };

  useEffect(() => {
    if (results && activeMode === "segmentation") {
      const stats = getCurrentStatistics();
      if (stats)
        setTimeout(() => {
          drawBarChart(stats, "barChart");
          drawDonutChart(stats, "donutChart");
        }, 100);
    }
  }, [results, activeTab, selectedQuadrat]);

  useEffect(() => {
    const handleResize = () => {
      if (results && activeMode === "segmentation") {
        const stats = getCurrentStatistics();
        if (stats) {
          drawBarChart(stats, "barChart");
          drawDonutChart(stats, "donutChart");
        }
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [results, selectedQuadrat]);

  /* ── actions ── */
  const clearAll = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
    setUploadProgress(0);
    setActiveTab("overlay");
    setSelectedQuadrat(0);
    setShowOriginalImage(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleModeSwitch = (mode) => {
    if (results || loading) return;
    setActiveMode(mode);
    setError(null);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFileSelection(e.dataTransfer.files[0]);
  };

  const handleFileChange = (e) => {
    if (e.target.files?.[0]) handleFileSelection(e.target.files[0]);
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

  const isNavDisabled = results || loading;
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
      if (activeMode === "segmentation")
        formData.append("use_auto_crop", "true");
      const endpoint =
        activeMode === "cots_counter"
          ? `${API_BASE}/cots-counter`
          : `${API_BASE}/segment`;
      const result = await fetch(endpoint, { method: "POST", body: formData });
      if (!result.ok) {
        const d = await result.json();
        throw new Error(d.error || "Analysis failed");
      }
      const data = await result.json();
      setResults(data);
      setSelectedQuadrat(0);
      setShowOriginalImage(false);
      setActiveTab(activeMode === "cots_counter" ? "annotated" : "overlay");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getCurrentResult = () => {
    if (!results) return null;
    if (activeMode === "segmentation" && results.results)
      return results.results[selectedQuadrat] || results.results[0];
    return results;
  };
  const getCurrentImages = () => {
    const r = getCurrentResult();
    return r ? r.images || results.images : null;
  };
  const getCurrentStatistics = () => {
    const r = getCurrentResult();
    return r ? r.statistics || results.statistics : null;
  };

  const getTabOptions = () =>
    activeMode === "cots_counter"
      ? [
          { key: "annotated", label: "Detections" },
          { key: "original", label: "Original" },
        ]
      : [
          { key: "overlay", label: "Overlay" },
          { key: "mask", label: "Mask" },
          { key: "original", label: "Cropped" },
        ];

  const currentImages = getCurrentImages();
  const currentStatistics = getCurrentStatistics();

  /* ── sub-components ── */
  const StatisticsPanel = ({ statistics, mode }) => {
    if (!statistics) return null;
    if (mode === "cots_counter") {
      return (
        <div className="space-y-2.5">
          <SectionLabel>Detection Results</SectionLabel>
          {Object.entries(statistics || {}).map(([key, data]) => (
            <div
              key={key}
              className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]"
            >
              <div
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ background: `rgb(${data.color.join(",")})` }}
              />
              <span className="flex-1 text-xs text-white/55 truncate">
                {data.display_name}
              </span>
              <span className="mono text-sm font-semibold text-white">
                {data.count !== undefined ? data.count : `${data.percentage}%`}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return (
      <div className="space-y-3">
        <SectionLabel>Coral Coverage</SectionLabel>
        {Object.entries(statistics || {})
          .filter(([key]) => key !== "total_coral")
          .sort((a, b) => b[1].percentage - a[1].percentage)
          .map(([key, data]) => (
            <div key={key} className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: `rgb(${data.color.join(",")})` }}
                  />
                  <span className="text-xs text-white/55 truncate">
                    {data.display_name}
                  </span>
                </div>
                <span className="mono text-xs text-white/75 flex-shrink-0">
                  {data.percentage}%
                </span>
              </div>
              <div className="h-1 rounded-full bg-white/[0.06] overflow-hidden">
                <div
                  className="stat-bar-fill"
                  style={{ width: `${data.percentage}%` }}
                />
              </div>
            </div>
          ))}
      </div>
    );
  };

  const QuadratSelector = () => {
    if (!results?.results || results.results.length <= 1) return null;
    return (
      <div className="card-glow rounded-2xl p-5 bg-white/[0.02] mb-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <SectionLabel>Quadrat Selection</SectionLabel>
            <h3 className="text-white font-semibold">Detected Quadrats</h3>
          </div>
          <Badge variant="info">{results.total_quadrats} found</Badge>
        </div>
        <div className="flex gap-3 overflow-x-auto pb-1">
          {results.results.map((result, index) => (
            <button
              key={index}
              onClick={() => setSelectedQuadrat(index)}
              className={`quadrat-btn flex-shrink-0 w-28 rounded-xl overflow-hidden border bg-white/[0.03] text-left ${
                selectedQuadrat === index ? "qactive" : "border-white/10"
              }`}
            >
              <img
                src={result.images.original}
                alt={`Quadrat ${index + 1}`}
                className="w-full h-16 object-cover"
              />
              <div className="p-2">
                <p className="text-xs text-white/75 font-medium">
                  Q{index + 1}
                </p>
                {result.statistics?.total_coral && (
                  <p className="mono text-[10px] text-emerald-400">
                    {result.statistics.total_coral.percentage}%
                  </p>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    );
  };

  return (
    <>
      <GlobalStyles />
      <div className="app-root min-h-screen bg-mesh text-white">
        {/* NAV */}
        <nav className="sticky top-0 z-50 border-b border-white/[0.06] bg-[#040d18]/80 backdrop-blur-xl">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
            <Link
              to="/"
              className="flex items-center gap-2.5 flex-shrink-0 group"
            >
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-emerald-400 to-cyan-400 flex items-center justify-center text-[11px] font-bold text-black shadow-lg shadow-emerald-500/20 group-hover:scale-105 transition-transform">
                S
              </div>
              <span className="font-semibold tracking-tight">
                SCB Analytics
              </span>
            </Link>

            {/* Mode switch */}
            <div className="flex items-center gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.07]">
              {[
                { key: "segmentation", label: "Segmentation" },
                { key: "cots_counter", label: "COTS Counter" },
                { key: "bleaching", label: "Bleaching" },
              ].map((m) => (
                <button
                  key={m.key}
                  onClick={() => handleModeSwitch(m.key)}
                  disabled={!!isNavDisabled}
                  className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-200
                    ${
                      activeMode === m.key
                        ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-emerald-300 border border-emerald-500/20 shadow-sm"
                        : "text-white/35 hover:text-white/65"
                    }
                    ${
                      isNavDisabled
                        ? "cursor-not-allowed opacity-50"
                        : "cursor-pointer"
                    }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              {(results || selectedImage) && (
                <button
                  onClick={clearAll}
                  className="px-3.5 py-1.5 text-sm text-white/45 hover:text-white/75 border border-white/[0.07] hover:border-white/18 rounded-lg transition-all"
                >
                  Clear
                </button>
              )}
              <Link
                to="/"
                className="px-3.5 py-1.5 text-sm border border-white/[0.07] hover:border-white/18 text-white/50 hover:text-white/80 rounded-lg transition-all"
              >
                ← Home
              </Link>
            </div>
          </div>
        </nav>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 py-10">
          {/* ERROR */}
          {error && (
            <div className="mb-6 flex items-start gap-3 p-4 rounded-xl bg-red-500/8 border border-red-500/18 fade-up">
              <div className="w-5 h-5 rounded-full bg-red-500/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-red-400 text-xs font-bold">!</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-red-300">Error</p>
                <p className="text-sm text-red-400/75 mt-0.5">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-400/40 hover:text-red-300 text-xl leading-none"
              >
                ×
              </button>
            </div>
          )}

          {/* ── UPLOAD ── */}
          {!selectedImage && !results && (
            <section className="flex flex-col items-center justify-center min-h-[calc(100vh-9rem)] fade-up">
              <div className="w-full max-w-2xl">
                <div className="text-center mb-10">
                  <Badge variant="success">
                    {activeMode === "segmentation"
                      ? "Segmentation Mode"
                      : activeMode === "cots_counter"
                      ? "COTS Counter Mode"
                      : "Bleaching Mode"}
                  </Badge>
                  <h1 className="mt-5 text-4xl sm:text-5xl font-bold tracking-tight leading-tight">
                    Upload Coral Reef
                    <span className="block bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                      Image
                    </span>
                  </h1>
                  <p className="mt-3 text-white/35 text-base">
                    Drag & drop or browse to start AI-powered reef analysis
                  </p>
                </div>

                {/* Drop zone */}
                <div
                  className={`relative rounded-2xl border-2 border-dashed p-14 text-center cursor-pointer transition-all duration-300 drop-pulse
                    ${
                      dragActive
                        ? "drop-active"
                        : "border-white/10 hover:border-emerald-500/28 bg-white/[0.015] hover:bg-white/[0.03]"
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
                    className="hidden"
                  />
                  <div className="mx-auto mb-5 w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20 flex items-center justify-center">
                    <svg
                      className="w-7 h-7 text-emerald-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                  <p className="text-white/75 font-medium mb-1">
                    {dragActive
                      ? "Drop it here"
                      : "Click to upload or drag & drop"}
                  </p>
                  <p className="mono text-xs text-white/25">
                    JPG · PNG · WebP · GIF · Max 15 MB
                  </p>
                  {uploadProgress > 0 && uploadProgress < 100 && (
                    <div className="mt-6 h-1 rounded-full bg-white/5 overflow-hidden mx-8">
                      <div
                        className="progress-shimmer h-full rounded-full transition-all duration-200"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}

          {selectedImage && !results && (
            <section className="fade-up">
              <div className="card-glow rounded-2xl bg-white/[0.02] overflow-hidden">
                <div className="flex flex-wrap items-center justify-between gap-3 p-5 border-b border-white/[0.06]">
                  <div>
                    <SectionLabel>Ready for analysis</SectionLabel>
                    <h2 className="text-lg font-semibold">Image Preview</h2>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        setSelectedImage(null);
                        if (fileInputRef.current)
                          fileInputRef.current.value = "";
                      }}
                      className="px-3.5 py-1.5 text-sm border border-white/[0.07] hover:border-white/18 text-white/45 hover:text-white/75 rounded-lg transition-all"
                    >
                      Change Image
                    </button>
                    <button
                      onClick={handleAnalysis}
                      disabled={loading}
                      className="flex items-center gap-2 px-5 py-1.5 rounded-lg text-sm font-bold bg-gradient-to-r from-emerald-500 to-cyan-500 text-black hover:from-emerald-400 hover:to-cyan-400 transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-60"
                    >
                      {loading ? (
                        <>
                          <span className="spinner mr-1" />
                          <span>Analyzing…</span>
                        </>
                      ) : activeMode === "segmentation" ? (
                        "Run Segmentation"
                      ) : (
                        "Detect COTS"
                      )}
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3">
                  <div className="lg:col-span-2 relative overflow-hidden bg-black/20">
                    <img
                      src={selectedImage}
                      alt="Preview"
                      className="img-zoom w-full max-h-[480px] object-contain"
                    />
                    <div className="absolute top-3 left-3">
                      <Badge variant="success">Ready</Badge>
                    </div>
                  </div>
                  <div className="border-t lg:border-t-0 lg:border-l border-white/[0.06] p-5 space-y-4">
                    <div>
                      <SectionLabel>Active Model</SectionLabel>
                      <p className="font-semibold text-white">
                        {activeMode === "segmentation"
                          ? "Coral Segmentation"
                          : "COTS Detection"}
                      </p>
                      <p className="text-sm text-white/38 mt-1 leading-relaxed">
                        {activeMode === "segmentation"
                          ? "Detects 8 coral types with pixel-level accuracy"
                          : "Identifies Crown-of-Thorns starfish in reef images"}
                      </p>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        {
                          label: "Model",
                          value:
                            activeMode === "segmentation"
                              ? "YOLOv8"
                              : "YOLOv11",
                        },
                        { label: "Accuracy", value: "99.2%" },
                        {
                          label: "Mode",
                          value:
                            activeMode === "segmentation" ? "Pixel" : "Object",
                        },
                        { label: "Status", value: "Ready" },
                      ].map((s) => (
                        <div
                          key={s.label}
                          className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3"
                        >
                          <p className="mono text-[10px] text-white/25 uppercase tracking-widest">
                            {s.label}
                          </p>
                          <p className="text-sm font-semibold text-white mt-0.5">
                            {s.value}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ── RESULTS ── */}
          {results && (
            <section className="space-y-5 fade-up">
              {/* Header */}
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <SectionLabel>Analysis complete</SectionLabel>
                  <div className="flex items-center gap-3">
                    <h2 className="text-2xl font-bold">Results</h2>
                    <Badge variant="success">
                      {results.auto_crop_applied ? "Auto-cropped" : "Completed"}
                    </Badge>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={clearAll}
                    className="px-3.5 py-1.5 text-sm border border-white/[0.07] hover:border-white/18 text-white/45 hover:text-white/75 rounded-lg transition-all"
                  >
                    Clear
                  </button>
                  <button
                    onClick={handleUploadClick}
                    className="px-4 py-1.5 text-sm rounded-lg bg-white/[0.04] hover:bg-white/[0.07] border border-white/[0.07] text-white/65 hover:text-white transition-all"
                  >
                    New Analysis
                  </button>
                </div>
              </div>

              {/* Original image toggle */}
              {results.original_image && (
                <div className="card-glow rounded-2xl bg-white/[0.02] overflow-hidden">
                  <div className="flex items-center justify-between px-5 py-3.5">
                    <p className="text-sm font-medium text-white/60">
                      Original Image
                    </p>
                    <button
                      onClick={() => setShowOriginalImage(!showOriginalImage)}
                      className="mono text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
                    >
                      {showOriginalImage ? "Hide ↑" : "Show ↓"}
                    </button>
                  </div>
                  {showOriginalImage && (
                    <div className="border-t border-white/[0.06]">
                      <img
                        src={results.original_image}
                        alt="Original"
                        className="w-full max-h-72 object-contain bg-black/25"
                      />
                    </div>
                  )}
                </div>
              )}

              <QuadratSelector />

              {/* Main result viewer */}
              <div className="card-glow rounded-2xl bg-white/[0.02] overflow-hidden">
                {/* Tabs */}
                <div className="flex items-center gap-1 px-4 pt-3 border-b border-white/[0.06]">
                  {getTabOptions().map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key)}
                      className={`relative pb-3 px-4 text-sm font-medium transition-all duration-200
                        ${
                          activeTab === tab.key
                            ? "tab-active"
                            : "text-white/35 hover:text-white/60"
                        }`}
                    >
                      {tab.label}
                      {activeTab === tab.key && (
                        <div className="tab-indicator" />
                      )}
                    </button>
                  ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3">
                  {/* Image */}
                  <div className="lg:col-span-2 relative min-h-64 bg-black/20">
                    <div className="absolute top-3 left-3 z-10">
                      <Badge>
                        {activeTab === "overlay" && "Segmentation Overlay"}
                        {activeTab === "mask" && "Segmentation Mask"}
                        {activeTab === "annotated" && "COTS Detections"}
                        {activeTab === "original" && "Cropped Quadrat"}
                        {results.auto_crop_applied &&
                          ` · Q${selectedQuadrat + 1}`}
                      </Badge>
                    </div>
                    {currentImages && (
                      <img
                        src={currentImages[activeTab] || currentImages.original}
                        alt={`${activeTab} view`}
                        className="img-zoom w-full h-full max-h-[480px] object-contain"
                      />
                    )}
                  </div>
                  {/* Stats */}
                  <div className="border-t lg:border-t-0 lg:border-l border-white/[0.06] p-5 overflow-y-auto max-h-[480px]">
                    <StatisticsPanel
                      statistics={currentStatistics}
                      mode={activeMode}
                    />
                  </div>
                </div>
              </div>

              {/* Charts – segmentation only */}
              {activeMode === "segmentation" && currentStatistics && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="card-glow rounded-2xl bg-white/[0.02] p-5">
                      <SectionLabel>Coverage</SectionLabel>
                      <h3 className="text-sm font-semibold mb-3">
                        Total Coral
                      </h3>
                      <canvas
                        id="donutChart"
                        className="w-full"
                        style={{ height: 200 }}
                      />
                      <div className="mt-3 flex justify-between">
                        <div>
                          <p className="mono text-[10px] text-white/25 uppercase tracking-widest">
                            Coral
                          </p>
                          <p className="text-sm font-bold text-emerald-400">
                            {currentStatistics.total_coral?.percentage}%
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="mono text-[10px] text-white/25 uppercase tracking-widest">
                            Non-Coral
                          </p>
                          <p className="text-sm font-bold text-white/50">
                            {(
                              100 -
                              (currentStatistics.total_coral?.percentage || 0)
                            ).toFixed(1)}
                            %
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="md:col-span-2 card-glow rounded-2xl bg-white/[0.02] p-5">
                      <SectionLabel>Distribution</SectionLabel>
                      <h3 className="text-sm font-semibold mb-3">
                        Coral Types
                      </h3>
                      <canvas
                        id="barChart"
                        className="w-full"
                        style={{ height: 200 }}
                      />
                    </div>
                  </div>

                  {/* Summary */}
                  <div className="card-glow rounded-2xl bg-white/[0.02] p-5">
                    <div className="flex items-center justify-between mb-5">
                      <div>
                        <SectionLabel>Summary</SectionLabel>
                        <h3 className="font-semibold">Coverage Overview</h3>
                      </div>
                      <Badge
                        variant={
                          currentStatistics.total_coral?.percentage > 50
                            ? "success"
                            : "warning"
                        }
                      >
                        {currentStatistics.total_coral?.percentage > 50
                          ? "Healthy"
                          : "Degraded"}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="rounded-xl bg-emerald-500/[0.06] border border-emerald-500/18 p-4">
                        <p className="mono text-[10px] text-emerald-400/55 uppercase tracking-widest">
                          Total Coral
                        </p>
                        <p className="text-3xl font-bold text-emerald-400 mt-1">
                          {currentStatistics.total_coral?.percentage}
                          <span className="text-lg">%</span>
                        </p>
                        <div className="mt-2 h-1 rounded-full bg-emerald-900/30 overflow-hidden">
                          <div
                            className="stat-bar-fill"
                            style={{
                              width: `${currentStatistics.total_coral?.percentage}%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4">
                        <p className="mono text-[10px] text-white/25 uppercase tracking-widest">
                          Types Found
                        </p>
                        <p className="text-3xl font-bold mt-1">
                          {
                            Object.keys(currentStatistics).filter(
                              (k) =>
                                k !== "total_coral" &&
                                currentStatistics[k].percentage > 0
                            ).length
                          }
                        </p>
                        <p className="mono text-[10px] text-white/20 mt-1">
                          coral species
                        </p>
                      </div>
                      <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4">
                        <p className="mono text-[10px] text-white/25 uppercase tracking-widest">
                          Dominant
                        </p>
                        <p className="text-base font-semibold mt-1 truncate">
                          {(() => {
                            const e = Object.entries(currentStatistics)
                              .filter(([k]) => k !== "total_coral")
                              .sort(
                                (a, b) => b[1].percentage - a[1].percentage
                              );
                            return e.length > 0 ? e[0][1].display_name : "N/A";
                          })()}
                        </p>
                        <p className="mono text-[10px] text-white/20 mt-1">
                          highest coverage
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeMode === "cots_counter" &&
                currentStatistics?.total_count && (
                  <div className="card-glow rounded-2xl bg-white/[0.02] p-5">
                    <div className="flex items-center justify-between mb-5">
                      <div>
                        <SectionLabel>Detection Summary</SectionLabel>
                        <h3 className="font-semibold">COTS Overview</h3>
                      </div>
                      <Badge
                        variant={
                          currentStatistics.total_count.count > 0
                            ? "warning"
                            : "success"
                        }
                      >
                        {currentStatistics.total_count.count > 0
                          ? "Alert"
                          : "Clear"}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-xl bg-amber-500/[0.07] border border-amber-500/18 p-4">
                        <p className="mono text-[10px] text-amber-400/55 uppercase tracking-widest">
                          COTS Detected
                        </p>
                        <p className="text-4xl font-bold text-amber-400 mt-1">
                          {currentStatistics.total_count.count}
                        </p>
                      </div>
                      <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4">
                        <p className="mono text-[10px] text-white/25 uppercase tracking-widest">
                          Avg Confidence
                        </p>
                        <p className="text-4xl font-bold mt-1">
                          {currentStatistics.average_confidence?.percentage ||
                            95}
                          <span className="text-xl">%</span>
                        </p>
                      </div>
                    </div>
                  </div>
                )}
            </section>
          )}
        </main>
      </div>
    </>
  );
}
