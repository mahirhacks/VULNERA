import { useCallback, useEffect, useState } from "react";
import { browseDirectory } from "../api/client";

export default function DirectoryBrowserModal({
  open,
  initialPath,
  onClose,
  onSelect,
  browseFn,
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [view, setView] = useState(null);

  const loadBrowse = useCallback(async (path) => {
    setLoading(true);
    setError(null);
    try {
      const requestBrowse = browseFn || browseDirectory;
      const result = await requestBrowse(path);
      setView(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [browseFn]);

  useEffect(() => {
    if (!open) return;
    loadBrowse(initialPath || undefined);
  }, [open, initialPath, loadBrowse]);

  if (!open) return null;

  const handleSelectCurrent = () => {
    if (view?.mode === "directory" && view.storage_path) {
      onSelect({
        storage_path: view.storage_path,
        current_path: view.current_path,
      });
      onClose();
    }
  };

  const handleOpenRoot = (path) => {
    loadBrowse(path);
  };

  const handleOpenFolder = (path) => {
    loadBrowse(path);
  };

  return (
    <div className="dir-browser-backdrop" onClick={onClose} role="presentation">
      <div
        className="dir-browser-panel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Browse models directory"
      >
        <div className="dir-browser-header">
          <h2>Select models directory</h2>
          <button type="button" className="dir-browser-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {error && <div className="dir-browser-error">{error}</div>}

        {loading && <div className="dir-browser-loading">Loading…</div>}

        {!loading && view?.mode === "roots" && (
          <div className="dir-browser-body">
            <p className="dir-browser-hint">Choose a starting location, then open the folder that contains your model checkpoints.</p>
            <ul className="dir-browser-list">
              {view.roots.map((root) => (
                <li key={root.path}>
                  <button type="button" className="dir-browser-item" onClick={() => handleOpenRoot(root.path)}>
                    <span className="dir-browser-folder-icon" aria-hidden="true" />
                    <span className="dir-browser-item-label">{root.label}</span>
                    <span className="dir-browser-item-path">{root.path}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {!loading && view?.mode === "directory" && (
          <div className="dir-browser-body">
            <div className="dir-browser-toolbar">
              {view.parent_path ? (
                <button type="button" className="dir-browser-up" onClick={() => loadBrowse(view.parent_path)}>
                  ↑ Up
                </button>
              ) : (
                <button type="button" className="dir-browser-up" onClick={() => loadBrowse()}>
                  ↑ Locations
                </button>
              )}
              <div className="dir-browser-current" title={view.current_path}>
                {view.current_path}
              </div>
            </div>

            {view.entries.length === 0 ? (
              <p className="dir-browser-empty">No subfolders here.</p>
            ) : (
              <ul className="dir-browser-list">
                {view.entries.map((entry) => (
                  <li key={entry.path}>
                    <button
                      type="button"
                      className="dir-browser-item"
                      onClick={() => handleOpenFolder(entry.path)}
                    >
                      <span className="dir-browser-folder-icon" aria-hidden="true" />
                      <span className="dir-browser-item-label">{entry.name}</span>
                      {entry.has_children ? <span className="dir-browser-chevron" aria-hidden="true">›</span> : null}
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <div className="dir-browser-footer">
              <button type="button" className="settings-btn primary" onClick={handleSelectCurrent}>
                Use this folder
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
