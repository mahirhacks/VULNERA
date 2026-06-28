import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  createProject,
  deleteProject,
  deleteScan,
  listProjects,
  listStandaloneScans,
} from "../api/client";
import CreateProjectModal from "./CreateProjectModal";
import HistoryContextMenu from "./HistoryContextMenu";
import "../styles/sidebar.css";

function SettingsIcon() {
  return (
    <svg
      className="sidebar-nav-icon"
      viewBox="0 0 20 20"
      aria-hidden="true"
      fill="none"
    >
      <path
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        d="M4 6.5h12M4 13.5h12"
      />
      <path
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M7.25 4.75a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5ZM12.75 11.75a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5Z"
      />
    </svg>
  );
}

function NewScanIcon() {
  return (
    <svg className="sidebar-action-icon" viewBox="0 0 20 20" aria-hidden="true" fill="none">
      <path
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M5.25 3.75h6.5l3 3v8.5a1 1 0 0 1-1 1h-8.5a1 1 0 0 1-1-1V4.75a1 1 0 0 1 1-1Z"
      />
      <path
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M11.75 3.75v3h3M7 11h6M10 8v6"
      />
    </svg>
  );
}

function NewProjectIcon() {
  return (
    <svg className="sidebar-action-icon" viewBox="0 0 20 20" aria-hidden="true" fill="none">
      <path
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3.25 6.25a1.5 1.5 0 0 1 1.5-1.5h3.1l1.4 1.5h6a1.5 1.5 0 0 1 1.5 1.5v6.5a1.5 1.5 0 0 1-1.5 1.5H4.75a1.5 1.5 0 0 1-1.5-1.5v-8Z"
      />
      <path
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
        d="M10 9.25v4M8 11.25h4"
      />
    </svg>
  );
}

function SidebarToggleIcon({ direction = "collapse" }) {
  const isExpand = direction === "expand";
  return (
    <svg className="sidebar-toggle-icon" viewBox="0 0 20 20" aria-hidden="true" fill="none">
      <path
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4.25 4.25h11.5v11.5H4.25zM8 4.25v11.5"
      />
      <path
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
        strokeLinejoin="round"
        d={isExpand ? "M11.25 7.25 14 10l-2.75 2.75" : "M14 7.25 11.25 10 14 12.75"}
      />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg className="project-folder-icon" viewBox="0 0 16 16" aria-hidden="true">
      <path
        fill="currentColor"
        d="M1.5 3.5A1.5 1.5 0 0 1 3 2h3.17a1.5 1.5 0 0 1 1.06.44L8.5 3.7H13a1.5 1.5 0 0 1 1.5 1.5v7A1.5 1.5 0 0 1 13 13.5H3A1.5 1.5 0 0 1 1.5 12V3.5z"
      />
    </svg>
  );
}

function SectionChevron() {
  return (
    <svg className="sidebar-section-chevron" viewBox="0 0 16 16" aria-hidden="true">
      <path
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4 6l4 4 4-4"
      />
    </svg>
  );
}

function SidebarSection({ title, open, onToggle, spaced = false, children }) {
  return (
    <section className={`sidebar-section${spaced ? " spaced" : ""}`}>
      <button
        type="button"
        className="sidebar-section-header"
        onClick={onToggle}
        aria-expanded={open}
      >
        <span className="sidebar-section-title-text">{title}</span>
        <SectionChevron />
      </button>
      <div className={`sidebar-collapse-panel${open ? " open" : ""}`}>
        <div className="sidebar-collapse-panel-inner">{children}</div>
      </div>
    </section>
  );
}

function ProjectChildren({ open, children }) {
  return (
    <div className={`sidebar-collapse-panel project-children-panel${open ? " open" : ""}`}>
      <div className="sidebar-collapse-panel-inner project-children">{children}</div>
    </div>
  );
}

export default function Sidebar({ collapsed = false, onToggle, refreshKey = 0, onRefresh }) {
  const [projects, setProjects] = useState([]);
  const [scans, setScans] = useState([]);
  const [projectExpanded, setProjectExpanded] = useState({});
  const [sectionsOpen, setSectionsOpen] = useState({ projects: true, pastUploads: true });
  const [menuOpenId, setMenuOpenId] = useState(null);
  const [menuType, setMenuType] = useState(null);
  const [createOpen, setCreateOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const menuRef = useRef(null);
  const menuPanelRef = useRef(null);

  const activeScanId = location.pathname.startsWith("/scan/")
    ? location.pathname.split("/")[2]
    : null;
  const activeProjectId = location.pathname.startsWith("/project/")
    ? location.pathname.split("/")[2]
    : null;

  const loadSidebar = () => {
    Promise.all([listProjects(), listStandaloneScans()])
      .then(([projectRows, scanRows]) => {
        setProjects(projectRows);
        setScans(scanRows);
        setProjectExpanded((current) => {
          const next = { ...current };
          for (const project of projectRows) {
            if (next[project.project_id] === undefined) {
              next[project.project_id] = true;
            }
          }
          return next;
        });
      })
      .catch(console.error);
  };

  useEffect(() => {
    loadSidebar();
  }, [refreshKey, location.pathname]);

  useEffect(() => {
    if (!menuOpenId) return;
    const close = (e) => {
      if (menuRef.current?.contains(e.target) || menuPanelRef.current?.contains(e.target)) {
        return;
      }
      setMenuOpenId(null);
      setMenuType(null);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuOpenId]);

  const handleNewScan = () => navigate("/");

  const toggleProject = (projectId) => {
    setProjectExpanded((current) => ({
      ...current,
      [projectId]: !current[projectId],
    }));
  };

  const toggleSection = (section) => {
    setSectionsOpen((current) => ({
      ...current,
      [section]: !current[section],
    }));
  };

  const openMenu = (type, id, event) => {
    event.preventDefault();
    event.stopPropagation();
    setMenuType(type);
    setMenuOpenId(id);
  };

  const handleDeleteScan = async (scanId) => {
    setMenuOpenId(null);
    setMenuType(null);
    try {
      await deleteScan(scanId);
      if (activeScanId === scanId) navigate("/");
      loadSidebar();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteProject = async (projectId) => {
    setMenuOpenId(null);
    setMenuType(null);
    try {
      await deleteProject(projectId);
      if (activeProjectId === projectId) navigate("/");
      onRefresh?.();
      loadSidebar();
    } catch (err) {
      console.error(err);
    }
  };

  const handleProjectCreated = async (name) => createProject(name);

  const handleCreateClose = (project) => {
    setCreateOpen(false);
    if (!project?.project_id) {
      return;
    }
    setProjects((current) => {
      if (current.some((row) => row.project_id === project.project_id)) {
        return current;
      }
      return [
        ...current,
        {
          project_id: project.project_id,
          name: project.name,
          created_at: project.created_at,
          updated_at: project.updated_at,
          scan_count: project.scan_count ?? 0,
          scans: project.scans ?? [],
        },
      ];
    });
    setProjectExpanded((current) => ({
      ...current,
      [project.project_id]: true,
    }));
    onRefresh?.();
    navigate(`/project/${project.project_id}`);
  };

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
      <div className="sidebar-inner">
      <div className="sidebar-top">
        <div className="sidebar-brand">
          <button
            type="button"
            className="sidebar-brand-toggle"
            onClick={collapsed ? onToggle : undefined}
            aria-label={collapsed ? "Expand sidebar" : "VULNERA"}
            title={collapsed ? "Expand sidebar" : "VULNERA"}
            tabIndex={collapsed ? 0 : -1}
          >
            <span className="sidebar-brand-title">VULNERA</span>
            <span className="sidebar-brand-initial">V</span>
            <span className="sidebar-brand-expand">
              <SidebarToggleIcon direction="expand" />
            </span>
          </button>
          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={onToggle}
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
          >
            <SidebarToggleIcon />
          </button>
        </div>

        <button type="button" className="sidebar-btn sidebar-action-btn" onClick={handleNewScan} title="New scan">
          <NewScanIcon />
          <span className="sidebar-btn-label">New scan</span>
        </button>
        <button
          type="button"
          className="sidebar-btn sidebar-action-btn"
          onClick={() => setCreateOpen(true)}
          title="New project"
        >
          <NewProjectIcon />
          <span className="sidebar-btn-label">New project</span>
        </button>
      </div>

      <div className="sidebar-history">
        <SidebarSection
          title="Projects"
          open={sectionsOpen.projects}
          onToggle={() => toggleSection("projects")}
        >
          {projects.length === 0 ? (
            <div className="empty-records">No projects yet.</div>
          ) : (
            <div className="project-list">
              {projects.map((project) => {
                const isExpanded = projectExpanded[project.project_id] !== false;
                const isActiveProject = activeProjectId === project.project_id;
                return (
                  <div key={project.project_id} className="project-group">
                    <div className={`project-row${isActiveProject ? " active" : ""}`}>
                      <Link
                        to={`/project/${project.project_id}`}
                        className="project-link"
                        title={project.name}
                        aria-expanded={isExpanded}
                        onClick={() => toggleProject(project.project_id)}
                      >
                        <FolderIcon />
                        <span className="project-link-label">{project.name}</span>
                      </Link>
                      <div
                        className="history-menu-wrap"
                        ref={
                          menuOpenId === project.project_id && menuType === "project"
                            ? menuRef
                            : null
                        }
                      >
                        <button
                          type="button"
                          className="history-menu-btn"
                          aria-label={`Options for ${project.name}`}
                          aria-expanded={menuOpenId === project.project_id && menuType === "project"}
                          onClick={(e) => openMenu("project", project.project_id, e)}
                        >
                          ···
                        </button>
                        {menuOpenId === project.project_id && menuType === "project" && (
                          <HistoryContextMenu open anchorRef={menuRef} panelRef={menuPanelRef}>
                            <button
                              type="button"
                              className="history-dropdown-item danger"
                              role="menuitem"
                              onClick={() => handleDeleteProject(project.project_id)}
                            >
                              Delete project
                            </button>
                          </HistoryContextMenu>
                        )}
                      </div>
                    </div>
                    <ProjectChildren open={isExpanded}>
                      {project.scans?.length ? (
                        project.scans.map((scan) => (
                          <div
                            key={scan.scan_id}
                            className={`history-row project-child-row${
                              activeScanId === scan.scan_id ? " active" : ""
                            }`}
                          >
                            <Link
                              to={`/scan/${scan.scan_id}`}
                              className="history-link"
                              title={scan.filename}
                            >
                              {scan.filename}
                            </Link>
                            <div
                              className="history-menu-wrap"
                              ref={
                                menuOpenId === scan.scan_id && menuType === "scan"
                                  ? menuRef
                                  : null
                              }
                            >
                              <button
                                type="button"
                                className="history-menu-btn"
                                aria-label={`Options for ${scan.filename}`}
                                aria-expanded={menuOpenId === scan.scan_id && menuType === "scan"}
                                onClick={(e) => openMenu("scan", scan.scan_id, e)}
                              >
                                ···
                              </button>
                              {menuOpenId === scan.scan_id && menuType === "scan" && (
                                <HistoryContextMenu open anchorRef={menuRef} panelRef={menuPanelRef}>
                                  <button
                                    type="button"
                                    className="history-dropdown-item danger"
                                    role="menuitem"
                                    onClick={() => handleDeleteScan(scan.scan_id)}
                                  >
                                    Delete
                                  </button>
                                </HistoryContextMenu>
                              )}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="project-child-empty">No files yet</div>
                      )}
                    </ProjectChildren>
                  </div>
                );
              })}
            </div>
          )}
        </SidebarSection>

        <SidebarSection
          title="Past uploads"
          open={sectionsOpen.pastUploads}
          onToggle={() => toggleSection("pastUploads")}
          spaced
        >
          {scans.length === 0 ? (
            <div className="empty-records">No standalone uploads.</div>
          ) : (
            <div className="history-list">
              {scans.map((scan) => (
                <div
                  key={scan.scan_id}
                  className={`history-row${activeScanId === scan.scan_id ? " active" : ""}`}
                >
                  <Link
                    to={`/scan/${scan.scan_id}`}
                    className="history-link"
                    title={scan.filename}
                  >
                    {scan.filename}
                  </Link>
                  <div
                    className="history-menu-wrap"
                    ref={menuOpenId === scan.scan_id && menuType === "scan" ? menuRef : null}
                  >
                    <button
                      type="button"
                      className="history-menu-btn"
                      aria-label={`Options for ${scan.filename}`}
                      aria-expanded={menuOpenId === scan.scan_id && menuType === "scan"}
                      onClick={(e) => openMenu("scan", scan.scan_id, e)}
                    >
                      ···
                    </button>
                    {menuOpenId === scan.scan_id && menuType === "scan" && (
                      <HistoryContextMenu open anchorRef={menuRef} panelRef={menuPanelRef}>
                        <button
                          type="button"
                          className="history-dropdown-item danger"
                          role="menuitem"
                          onClick={() => handleDeleteScan(scan.scan_id)}
                        >
                          Delete
                        </button>
                      </HistoryContextMenu>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </SidebarSection>
      </div>

      <div className="sidebar-footer">
        <div className="sidebar-settings-divider" />
        <NavLink
          to="/settings"
          className={({ isActive }) => `sidebar-btn sidebar-nav${isActive ? " active" : ""}`}
        >
          <SettingsIcon />
          <span className="sidebar-btn-label">Settings</span>
        </NavLink>
      </div>

      <CreateProjectModal
        open={createOpen}
        onClose={handleCreateClose}
        onCreated={handleProjectCreated}
      />
      </div>
    </aside>
  );
}
