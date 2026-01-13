import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AppPage from "./components/AppPage";
import Home from "./components/Home";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

function HomePage() {
  return <Home />;
}

function ApplicationPage() {
  return <AppPage />;
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" exact element={<HomePage />} />
        <Route path="/application" element={<ApplicationPage />} />
      </Routes>
    </Router>
  );
}

export default App;
