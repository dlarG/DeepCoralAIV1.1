import React from "react";
import { Link } from "react-router-dom";
import "./LandingPage.css";

const Home = () => {
  return (
    <div className="landing-page">
      {/* Navigation Bar */}
      <nav className="landing-nav">
        <div className="nav-container">
          <div className="nav-logo">
            <span className="logo-text">SCB Marine Conserve</span>
          </div>
          <div className="nav-links">
            <a href="#features" className="nav-link">
              Features
            </a>
            <a href="#how-it-works" className="nav-link">
              How it Works
            </a>
            <a href="#stats" className="nav-link">
              Stats
            </a>
            <Link to="/application" className="nav-cta">
              Launch App
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-container">
          <div className="hero-content">
            <div className="hero-tag">
              <span>AI-Powered Marine Conservation</span>
            </div>
            <h1 className="hero-title">
              <span className="title-line">Advanced Coral Reef</span>
              <span className="title-line gradient-text">
                Intelligence Platform
              </span>
            </h1>
            <p className="hero-description">
              Transform underwater imagery into actionable insights with our
              state-of-the-art computer vision platform. Detect, segment, and
              monitor coral health with unprecedented accuracy.
            </p>
            <div className="hero-actions">
              <Link to="/application" className="btn-primary">
                <span className="btn-icon">🚀</span>
                Start Free Analysis
                <span className="btn-arrow">→</span>
              </Link>
              <a href="#demo" className="btn-outline">
                <span className="btn-icon">🎬</span>
                Watch Demo
              </a>
            </div>
          </div>
          <div className="hero-visual">
            <div className="visual-wrapper">
              <div className="main-visual">
                <div className="visual-overlay">
                  <div className="overlay-grid"></div>
                </div>
                <div className="floating-element element-1">
                  <div className="element-icon">🎯</div>
                  <div className="element-content">
                    <div className="element-title">Segmentation</div>
                    <div className="element-subtitle">8 classes</div>
                  </div>
                </div>
                <div className="floating-element element-2">
                  <div className="element-icon">⭐</div>
                  <div className="element-content">
                    <div className="element-title">COTS Detection</div>
                    <div className="element-subtitle">Real-time</div>
                  </div>
                </div>
                <div className="floating-element element-3">
                  <div className="element-icon">📊</div>
                  <div className="element-content">
                    <div className="element-title">Bleaching</div>
                    <div className="element-subtitle">Coral Bleaching</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section" id="features">
        <div className="section-container">
          <div className="section-header">
            <div className="section-subtitle">Core Capabilities</div>
            <h2 className="section-title">Comprehensive Reef Analysis</h2>
            <p className="section-description">
              Our platform combines multiple AI models to provide holistic reef
              monitoring
            </p>
          </div>
          <div className="features-grid">
            <div className="feature-card">
              <h3 className="feature-title">Pixel-Level Coral Segmentation</h3>
              <p className="feature-description">
                Advanced deep learning models identify and segment 8 distinct
                coral types with pixel-level precision for accurate coverage
                analysis.
              </p>
              <ul className="feature-list">
                <li>
                  <span className="list-icon">✓</span> Multi-class segmentation
                </li>
                <li>
                  <span className="list-icon">✓</span> Coverage statistics
                </li>
                <li>
                  <span className="list-icon">✓</span> Visual overlays
                </li>
              </ul>
            </div>

            <div className="feature-card feature-highlight">
              <h3 className="feature-title">COTS Detection & Counting</h3>
              <p className="feature-description">
                Real-time detection of Crown-of-Thorns starfish with bounding
                box annotations and individual counting for population
                monitoring.
              </p>
              <ul className="feature-list">
                <li>
                  <span className="list-icon">✓</span> YOLOv11 model
                </li>
                <li>
                  <span className="list-icon">✓</span> Confidence scoring
                </li>
                <li>
                  <span className="list-icon">✓</span> Batch processing
                </li>
              </ul>
            </div>

            <div className="feature-card">
              <h3 className="feature-title">Bleaching Assessment</h3>
              <p className="feature-description">
                Monitor coral health with automated bleaching detection and
                severity assessment tools for proactive conservation.
              </p>
              <ul className="feature-list">
                <li>
                  <span className="list-icon">🔜</span> Coming soon
                </li>
                <li>
                  <span className="list-icon">✓</span> Health metrics
                </li>
                <li>
                  <span className="list-icon">✓</span> Trend analysis
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="workflow-section" id="how-it-works">
        <div className="section-container">
          <div className="section-header">
            <div className="section-subtitle">Simple Workflow</div>
            <h2 className="section-title">How It Works</h2>
          </div>
          <div className="workflow-steps">
            <div className="workflow-step">
              <div className="step-number">01</div>
              <div className="step-content">
                <h3 className="step-title">Upload</h3>
                <p className="step-description">
                  Upload underwater reef images in any common format. Our
                  platform handles image preprocessing automatically.
                </p>
              </div>
            </div>
            <div className="step-connector">
              <div className="connector-line"></div>
              <div className="connector-arrow">→</div>
            </div>
            <div className="workflow-step">
              <div className="step-number">02</div>
              <div className="step-content">
                <h3 className="step-title">Analyze</h3>
                <p className="step-description">
                  Choose your analysis mode and let our AI models process the
                  image. Results are ready in seconds.
                </p>
              </div>
            </div>
            <div className="step-connector">
              <div className="connector-line"></div>
              <div className="connector-arrow">→</div>
            </div>
            <div className="workflow-step">
              <div className="step-number">03</div>
              <div className="step-content">
                <h3 className="step-title">Visualize</h3>
                <p className="step-description">
                  Explore interactive results with detailed statistics, and
                  visual overlays.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="demo-section">
        <div className="demo-container">
          <div className="demo-content">
            <div className="demo-header">
              <h2 className="demo-title">See It in Action</h2>
              <p className="demo-subtitle">
                Experience the power of AI-driven reef analysis with our
                interactive demo
              </p>
            </div>
            <div className="demo-visual">
              <div className="demo-frame">
                <div className="frame-header">
                  <div className="frame-dots">
                    <span className="dot red"></span>
                    <span className="dot yellow"></span>
                    <span className="dot green"></span>
                  </div>
                  <div className="frame-title">Live Analysis Preview</div>
                </div>
                <div className="frame-content">
                  <div className="preview-grid">
                    <div className="preview-item">
                      <div className="preview-image original"></div>
                      <div className="preview-label">Original</div>
                    </div>
                    <div className="preview-item">
                      <div className="preview-image segmented"></div>
                      <div className="preview-label">Segmented</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <Link to="/application" className="btn-primary btn-large">
              Try Live Demo
              <span className="btn-arrow">→</span>
            </Link>
          </div>
        </div>
      </section>
      <section className="cta-section">
        <div className="cta-container">
          <div className="cta-card">
            <div className="cta-content">
              <h2 className="cta-title">Start Your Free Analysis</h2>
              <p className="cta-description">
                Join marine biologists, researchers, and conservationists
                worldwide who trust our platform for accurate reef monitoring.
              </p>
              <div className="cta-actions">
                <Link to="/application" className="btn-primary btn-large">
                  <span className="btn-icon">🚀</span>
                  Launch Application
                  <span className="btn-arrow">→</span>
                </Link>
                <a href="#contact" className="btn-outline">
                  <span className="btn-icon">📞</span>
                  Contact Sales
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-container">
          <div className="footer-main">
            <div className="footer-brand">
              <div className="footer-logo">
                <span className="logo-text">SCB Marine Conserve</span>
              </div>
              <p className="footer-tagline">
                AI-powered solutions for marine conservation and reef monitoring
              </p>
            </div>
            <div className="footer-links">
              <div className="link-group">
                <h4 className="link-title">Product</h4>
                <a href="#features" className="link-item">
                  Features
                </a>
                <a href="#pricing" className="link-item">
                  Pricing
                </a>
                <a href="#demo" className="link-item">
                  Demo
                </a>
              </div>
              <div className="link-group">
                <h4 className="link-title">Resources</h4>
                <a href="#docs" className="link-item">
                  Documentation
                </a>
                <a href="#api" className="link-item">
                  API
                </a>
                <a href="#research" className="link-item">
                  Research Papers
                </a>
              </div>
              <div className="link-group">
                <h4 className="link-title">Company</h4>
                <a href="#about" className="link-item">
                  About
                </a>
                <a href="#contact" className="link-item">
                  Contact
                </a>
                <a href="#careers" className="link-item">
                  Careers
                </a>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <div className="footer-copyright">
              © 2024 SCB Marine Conserve. All rights reserved.
            </div>
            <div className="footer-social">
              <a href="#twitter" className="social-link">
                Twitter
              </a>
              <a href="#linkedin" className="social-link">
                LinkedIn
              </a>
              <a href="#github" className="social-link">
                GitHub
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;
