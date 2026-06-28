import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { startScan, streamJob } from "../api/client";
import PhaseProgress from "../components/PhaseProgress";
import "../styles/home.css";

const HOME_PROMPTS = [
  "Let's secure some C/C++ code.",
  "Upload code. Review risk. Move faster.",
  "What source file should we inspect today?",
  "Ready to scan your next C/C++ file.",
  "Drop in code and let VULNERA triage it.",
];

const ACCEPT = ".c,.cpp,.cc,.cxx,.h,.hpp";

export default function HomePage({ onScanComplete }) {
  const [prompt] = useState(() => HOME_PROMPTS[Math.floor(Math.random() * HOME_PROMPTS.length)]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  const onFileChange = (e) => {
    setSelectedFile(e.target.files?.[0] || null);
    setError(null);
  };

  const runScan = async () => {
    if (!selectedFile || job?.status === "running") return;
    setError(null);
    setJob(null);
    try {
      const { job_id } = await startScan(selectedFile);
      setJob({ job_id, status: "running", phase_index: 0, progress: 0 });
      streamJob(job_id, (snapshot) => {
        setJob(snapshot);
        if (snapshot.status === "completed" && snapshot.scan_id) {
          onScanComplete?.();
          navigate(`/scan/${snapshot.scan_id}`);
        }
        if (snapshot.status === "failed") {
          setError(snapshot.error || "Scan failed");
        }
      });
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="home-page">
      <div className="home-stack">
        <div className="home-eyebrow">VULNERA</div>
        <h1>{prompt}</h1>
        <p>Upload one C/C++ source file to start a focused vulnerability review.</p>

        <div className="upload-shell">
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="upload-input"
            onChange={onFileChange}
          />
          <button
            type="button"
            className="upload-btn"
            onClick={() => inputRef.current?.click()}
          >
            Upload C/C++ file
          </button>
        </div>
        <div className="upload-supported">Supported: .c, .cpp, .cc, .cxx, .h, .hpp</div>

        {selectedFile && (
          <div className="selected-file">Selected: {selectedFile.name}</div>
        )}
        {selectedFile && !job && (
          <button type="button" className="start-btn" onClick={runScan}>
            Start vulnerability scan
          </button>
        )}
        {selectedFile && job?.status === "failed" && (
          <button type="button" className="start-btn" onClick={runScan}>
            Retry scan
          </button>
        )}
        {error && <div className="home-error">{error}</div>}

        {job && job.status === "running" && <PhaseProgress job={job} />}
      </div>
    </div>
  );
}
