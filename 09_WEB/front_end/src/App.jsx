import { useCallback, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import HomePage from "./pages/HomePage";
import ProjectPage from "./pages/ProjectPage";
import ResultsPage from "./pages/ResultsPage";
import SettingsPage from "./pages/SettingsPage";
import "./styles/app.css";
import "./styles/severity.css";

const SIDEBAR_COLLAPSED_KEY = "vulnera.sidebarCollapsed";

function readSidebarCollapsed() {
  try {
    return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export default function App() {
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);
  const refreshSidebar = useCallback(() => {
    setSidebarRefreshKey((key) => key + 1);
  }, []);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(readSidebarCollapsed);

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((collapsed) => {
      const next = !collapsed;
      try {
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, next ? "1" : "0");
      } catch {
        // ignore storage errors
      }
      return next;
    });
  }, []);

  return (
    <BrowserRouter>
      <div className={`app-shell${sidebarCollapsed ? " sidebar-collapsed" : ""}`}>
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={toggleSidebar}
          refreshKey={sidebarRefreshKey}
          onRefresh={refreshSidebar}
        />
        <main className="main-panel">
          <Routes>
            <Route path="/" element={<HomePage onScanComplete={refreshSidebar} />} />
            <Route
              path="/project/:projectId"
              element={<ProjectPage onRefresh={refreshSidebar} />}
            />
            <Route path="/scan/:scanId" element={<ResultsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
