import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getProject, startScan, waitForJob } from "../api/client";
import PhaseProgress from "../components/PhaseProgress";
import "../styles/home.css";
import "../styles/project.css";

const ACCEPT = ".c,.cpp,.cc,.cxx,.h,.hpp";

export default function ProjectPage({ onRefresh }) {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const [project, setProject] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [job, setJob] = useState(null);
  const [batchProgress, setBatchProgress] = useState(null);
  const [error, setError] = useState(null);

  const loadProject = () => {
    getProject(projectId)
      .then(setProject)
      .catch((err) => {
        setError(err.message);
        navigate("/");
      });
  };

  useEffect(() => {
    loadProject();
  }, [projectId]);

  const onFilesChange = (event) => {
    const files = Array.from(event.target.files || []);
    setSelectedFiles(files);
    setError(null);
  };

  const runBatchScan = async () => {
    if (!selectedFiles.length || job?.status === "running") return;
    setError(null);
    setJob({ status: "running", progress: 0, detail: "Starting batch…" });
    setBatchProgress({ current: 0, total: selectedFiles.length, filename: "" });

    try {
      let lastScanId = null;
      for (let index = 0; index < selectedFiles.length; index += 1) {
        const file = selectedFiles[index];
        setBatchProgress({
          current: index + 1,
          total: selectedFiles.length,
          filename: file.name,
        });
        setJob({
          status: "running",
          progress: index / selectedFiles.length,
          detail: `Scanning ${file.name} (${index + 1}/${selectedFiles.length})`,
        });
        const { job_id } = await startScan(file, { projectId });
        const snapshot = await waitForJob(job_id);
        lastScanId = snapshot.scan_id;
      }
      setSelectedFiles([]);
      if (inputRef.current) inputRef.current.value = "";
      setJob(null);
      setBatchProgress(null);
      onRefresh?.();
      loadProject();
      if (lastScanId) {
        navigate(`/scan/${lastScanId}`);
      }
    } catch (err) {
      setJob({ status: "failed" });
      setError(err.message || "Batch scan failed.");
      setBatchProgress(null);
      loadProject();
    }
  };

  if (error && !project) {
    return <div className="project-page-error">{error}</div>;
  }
  if (!project) {
    return <div className="project-page-loading">Loading project…</div>;
  }

  return (
    <div className="project-page">
      <div className="project-page-stack">
        <div className="project-page-eyebrow">Project</div>
        <h1>{project.name}</h1>
        <p>
          Upload one or more C/C++ files to scan them into this project.
          Files are processed sequentially.
        </p>

        <div className="upload-shell">
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="upload-input"
            onChange={onFilesChange}
          />
          <button
            type="button"
            className="upload-btn"
            onClick={() => inputRef.current?.click()}
          >
            Add C/C++ files
          </button>
        </div>
        <div className="upload-supported">Supported: .c, .cpp, .cc, .cxx, .h, .hpp</div>

        {selectedFiles.length > 0 && (
          <div className="project-selected-files">
            <div className="project-selected-title">
              {selectedFiles.length} file{selectedFiles.length === 1 ? "" : "s"} selected
            </div>
            <ul className="project-selected-list">
              {selectedFiles.map((file) => (
                <li key={`${file.name}-${file.size}`}>{file.name}</li>
              ))}
            </ul>
          </div>
        )}

        {selectedFiles.length > 0 && !job && (
          <button type="button" className="start-btn" onClick={runBatchScan}>
            Scan {selectedFiles.length} file{selectedFiles.length === 1 ? "" : "s"}
          </button>
        )}
        {selectedFiles.length > 0 && job?.status === "failed" && (
          <button type="button" className="start-btn" onClick={runBatchScan}>
            Retry batch scan
          </button>
        )}
        {error && <div className="home-error">{error}</div>}
        {job?.status === "running" && (
          <PhaseProgress
            job={{
              ...job,
              phase_label: batchProgress
                ? `File ${batchProgress.current} of ${batchProgress.total}`
                : "Scanning",
              detail: batchProgress?.filename || job.detail,
            }}
          />
        )}

        <section className="project-files-section">
          <h2>Files in project</h2>
          {project.scans?.length ? (
            <ul className="project-files-list">
              {project.scans.map((scan) => (
                <li key={scan.scan_id}>
                  <Link to={`/scan/${scan.scan_id}`} className="project-file-link">
                    {scan.filename}
                  </Link>
                  <span className="project-file-meta">
                    {scan.function_count} fn · {scan.finding_count} findings
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="project-files-empty">No files scanned in this project yet.</p>
          )}
        </section>
      </div>
    </div>
  );
}
