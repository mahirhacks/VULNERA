import { useEffect, useRef, useState } from "react";
import {
  getHfModelVariants,
  getModelDownloadJob,
  startModelDownload,
  streamModelDownload,
} from "../api/client";
import PhaseProgress from "./PhaseProgress";

const FIT_CLASS = {
  not_recommended: "hf-fit-not_recommended",
  tight: "hf-fit-tight",
  acceptable: "hf-fit-acceptable",
  comfortable: "hf-fit-comfortable",
  high_speed: "hf-fit-high_speed",
};

function FitBadge({ fit }) {
  if (!fit) return null;
  return (
    <span className={`hf-fit-badge ${FIT_CLASS[fit.tier] || ""}`} title={fit.detail}>
      {fit.label}
    </span>
  );
}

export default function ModelDownloadModal({
  open,
  preset,
  modelsRootDir,
  onClose,
  onComplete,
}) {
  const [phase, setPhase] = useState("confirm");
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [variants, setVariants] = useState([]);
  const [selectedQuantId, setSelectedQuantId] = useState("q4_nf4");
  const completedRef = useRef(false);

  useEffect(() => {
    if (!open) {
      setPhase("confirm");
      setJob(null);
      setError(null);
      setVariants([]);
      setSelectedQuantId("q4_nf4");
      completedRef.current = false;
      return;
    }
    if (!preset?.repo_id) return;
    getHfModelVariants(preset.repo_id)
      .then((result) => {
        const modelVariants = result.model?.variants || [];
        setVariants(modelVariants);
        setSelectedQuantId(result.model?.recommended_variant_id || "q4_nf4");
      })
      .catch(() => {
        setVariants([]);
        setSelectedQuantId("q4_nf4");
      });
  }, [open, preset?.id, preset?.repo_id]);

  useEffect(() => {
    if (!open || phase !== "downloading" || !job?.job_id) return undefined;

    const stream = streamModelDownload(job.job_id, (snapshot) => {
      setJob(snapshot);
      if (snapshot.status === "completed" && !completedRef.current) {
        completedRef.current = true;
        setPhase("complete");
        onComplete?.({
          ...preset,
          quantization_id: snapshot.quantization_id || selectedQuantId,
        });
      }
      if (snapshot.status === "failed") {
        setPhase("failed");
        setError(snapshot.error || "Download failed");
      }
    });

    return () => stream.close();
  }, [open, phase, job?.job_id, onComplete, preset, selectedQuantId]);

  if (!open || !preset) return null;

  const selectedVariant = variants.find((item) => item.id === selectedQuantId) || variants[0] || null;

  const handleConfirm = async () => {
    setError(null);
    try {
      const { job_id } = await startModelDownload({
        models_root_dir: modelsRootDir,
        preset_id: preset.id,
        quantization_id: selectedQuantId,
      });
      const snapshot = await getModelDownloadJob(job_id);
      setJob(snapshot);
      setPhase("downloading");
    } catch (err) {
      setError(err.message);
      setPhase("failed");
    }
  };

  const targetFolder = `${modelsRootDir}/${preset.folder_name}`.replace(/\\/g, "/");

  return (
    <div className="dir-browser-backdrop" onClick={phase === "downloading" ? undefined : onClose} role="presentation">
      <div
        className="dir-browser-panel model-download-panel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={`Download ${preset.label}`}
      >
        <div className="dir-browser-header">
          <h2>{phase === "confirm" ? "Download model" : preset.label}</h2>
          {phase !== "downloading" && (
            <button type="button" className="dir-browser-close" onClick={onClose} aria-label="Close">
              ×
            </button>
          )}
        </div>

        <div className="model-download-body">
          {phase === "confirm" && (
            <>
              <p className="model-download-lead">
                <strong>{preset.label}</strong> is not in your models folder yet.
              </p>
              <p className="model-download-copy">
                Download from HuggingFace ({preset.repo_id}) into:
              </p>
              <code className="model-download-path">{targetFolder}</code>
              <p className="model-download-note">{preset.note}</p>

              {variants.length > 0 && (
                <div className="settings-field hf-quant-field">
                  <label className="settings-label" htmlFor="preset-quant-select">
                    Runtime quantization
                  </label>
                  <select
                    id="preset-quant-select"
                    className="settings-input"
                    value={selectedQuantId}
                    onChange={(e) => setSelectedQuantId(e.target.value)}
                  >
                    {variants.map((variant) => (
                      <option key={variant.id} value={variant.id}>
                        {variant.label}
                        {variant.needed_gb != null ? ` · ~${variant.needed_gb} GB needed` : ""}
                      </option>
                    ))}
                  </select>
                  {selectedVariant && (
                    <div className="hf-quant-detail">
                      <FitBadge fit={selectedVariant.fit} />
                      <p className="settings-hint">{selectedVariant.description}</p>
                      <p className="settings-hint">{selectedVariant.fit?.detail}</p>
                    </div>
                  )}
                </div>
              )}

              <p className="model-download-warning">
                This can take several GB and may take a while depending on your connection.
              </p>
              {error && <div className="dir-browser-error">{error}</div>}
              <div className="model-download-actions">
                <button type="button" className="settings-btn secondary" onClick={onClose}>
                  Cancel
                </button>
                <button type="button" className="settings-btn primary" onClick={handleConfirm}>
                  Download
                </button>
              </div>
            </>
          )}

          {phase === "downloading" && (
            <>
              <p className="model-download-copy">
                Downloading to your models directory
                {selectedVariant ? ` (${selectedVariant.short_label || selectedVariant.label})` : ""}…
              </p>
              <code className="model-download-path">{targetFolder}</code>
              <PhaseProgress job={job} />
            </>
          )}

          {phase === "complete" && (
            <>
              <p className="model-download-lead">Download complete.</p>
              <p className="model-download-copy">
                {preset.label} is ready in your models folder and has been selected.
              </p>
              <div className="model-download-actions">
                <button type="button" className="settings-btn primary" onClick={onClose}>
                  Done
                </button>
              </div>
            </>
          )}

          {phase === "failed" && (
            <>
              <p className="model-download-lead">Download failed</p>
              <div className="dir-browser-error">{error}</div>
              <div className="model-download-actions">
                <button type="button" className="settings-btn secondary" onClick={onClose}>
                  Close
                </button>
                <button type="button" className="settings-btn primary" onClick={() => setPhase("confirm")}>
                  Try again
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
