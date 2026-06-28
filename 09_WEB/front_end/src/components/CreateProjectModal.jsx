import { useState } from "react";
import "../styles/project.css";

export default function CreateProjectModal({ open, onClose, onCreated }) {
  const [name, setName] = useState("");
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  if (!open) return null;

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || saving) return;
    setSaving(true);
    setError(null);
    try {
      const project = await onCreated(trimmed);
      setName("");
      onClose(project);
    } catch (err) {
      setError(err.message || "Could not create project.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="project-modal-backdrop" onClick={() => onClose()}>
      <div
        className="project-modal"
        role="dialog"
        aria-labelledby="create-project-title"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="create-project-title">New project</h2>
        <p className="project-modal-copy">
          Group related C/C++ files together — like a ChatGPT project folder.
        </p>
        <form onSubmit={handleSubmit}>
          <label className="project-modal-label" htmlFor="project-name">
            Project name
          </label>
          <input
            id="project-name"
            className="project-modal-input"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Hotel module"
            autoFocus
            maxLength={120}
          />
          {error && <div className="project-modal-error">{error}</div>}
          <div className="project-modal-actions">
            <button type="button" className="project-modal-btn ghost" onClick={() => onClose()}>
              Cancel
            </button>
            <button type="submit" className="project-modal-btn primary" disabled={saving || !name.trim()}>
              {saving ? "Creating…" : "Create project"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
