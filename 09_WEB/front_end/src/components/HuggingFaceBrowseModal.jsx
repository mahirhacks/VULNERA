import { useCallback, useEffect, useRef, useState } from "react";
import {
  getHfConnectivity,
  getHfFilters,
  getHfModelVariants,
  getLlmHardware,
  searchHfModels,
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

export default function HuggingFaceBrowseModal({
  open,
  modelsRootDir,
  onClose,
  onComplete,
  pickOnly = false,
  onPick,
}) {
  const [phase, setPhase] = useState("browse");
  const [online, setOnline] = useState(true);
  const [filters, setFilters] = useState({ families: [], param_sizes: [], purposes: [] });
  const [query, setQuery] = useState("");
  const [family, setFamily] = useState("all");
  const [paramSize, setParamSize] = useState("any");
  const [purpose, setPurpose] = useState("code");
  const [page, setPage] = useState(0);
  const [models, setModels] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [selectedQuantId, setSelectedQuantId] = useState("q4_nf4");
  const [confirmVariants, setConfirmVariants] = useState([]);
  const [hardware, setHardware] = useState(null);
  const [recommendMode, setRecommendMode] = useState(false);
  const [job, setJob] = useState(null);
  const completedRef = useRef(false);

  const reset = useCallback(() => {
    setPhase("browse");
    setQuery("");
    setFamily("all");
    setParamSize("any");
    setPurpose("code");
    setPage(0);
    setModels([]);
    setHasMore(false);
    setError(null);
    setSelected(null);
    setSelectedQuantId("q4_nf4");
    setConfirmVariants([]);
    setHardware(null);
    setRecommendMode(false);
    setJob(null);
    completedRef.current = false;
  }, []);

  useEffect(() => {
    if (!open) {
      reset();
      return;
    }
    getHfConnectivity()
      .then((result) => setOnline(Boolean(result.online)))
      .catch(() => setOnline(false));
    getHfFilters()
      .then(setFilters)
      .catch(() => {});
  }, [open, reset]);

  const runSearch = useCallback(async (pageIndex = 0, append = false, withFit = recommendMode) => {
    if (!online) return;
    setLoading(true);
    setError(null);
    try {
      const result = await searchHfModels({
        query,
        family,
        param_size: paramSize,
        purpose,
        page: pageIndex,
        include_fit: withFit,
      });
      setModels((prev) => (append ? [...prev, ...result.models] : result.models));
      setHasMore(Boolean(result.has_more));
      setPage(result.page);
      if (result.hardware) {
        setHardware(result.hardware);
      }
    } catch (err) {
      setError(err.message);
      if (!append) setModels([]);
    } finally {
      setLoading(false);
    }
  }, [online, query, family, paramSize, purpose, recommendMode]);

  useEffect(() => {
    if (!open || phase !== "browse" || !online) return;
    runSearch(0, false, recommendMode);
  }, [open, phase, online, family, paramSize, purpose, recommendMode]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!open || phase !== "downloading" || !job?.job_id) return undefined;

    const stream = streamModelDownload(job.job_id, (snapshot) => {
      setJob(snapshot);
      if (snapshot.status === "completed" && !completedRef.current) {
        completedRef.current = true;
        setPhase("complete");
        onComplete?.({
          repo_id: selected.repo_id,
          label: selected.label,
          folder_name: snapshot.folder_name || selected.folder_name,
          quantization_id: snapshot.quantization_id || selectedQuantId,
        });
      }
      if (snapshot.status === "failed") {
        setPhase("failed");
        setError(snapshot.error || "Download failed");
      }
    });

    return () => stream.close();
  }, [open, phase, job?.job_id, onComplete, selected, selectedQuantId]);

  const handleRecommend = async () => {
    setError(null);
    try {
      const profile = hardware || await getLlmHardware();
      setHardware(profile);
      setRecommendMode(true);
    } catch (err) {
      setError(err.message);
    }
  };

  if (!open) return null;

  const handleSearchSubmit = (event) => {
    event.preventDefault();
    runSearch(0, false, recommendMode);
  };

  const handleSelectModel = async (model) => {
    if (pickOnly) {
      setError(null);
      onPick?.({
        repo_id: model.repo_id,
        label: model.label,
        folder_name: model.folder_name,
        param_label: model.param_label,
      });
      onClose();
      return;
    }

    if (!modelsRootDir?.trim()) {
      setError("Select a models directory with Browse before downloading.");
      return;
    }
    setError(null);
    setSelected(model);
    setSelectedQuantId(model.recommended_variant_id || "q4_nf4");
    if (model.variants?.length) {
      setConfirmVariants(model.variants);
      setPhase("confirm");
      return;
    }
    try {
      const result = await getHfModelVariants(model.repo_id);
      setHardware(result.hardware || hardware);
      setConfirmVariants(result.model?.variants || []);
      setSelectedQuantId(result.model?.recommended_variant_id || "q4_nf4");
      setPhase("confirm");
    } catch (err) {
      setError(err.message);
    }
  };

  const selectedVariant = confirmVariants.find((item) => item.id === selectedQuantId)
    || confirmVariants[0]
    || null;

  const handleConfirmDownload = async () => {
    if (!selected) return;
    setError(null);
    try {
      const { job_id } = await startModelDownload({
        models_root_dir: modelsRootDir,
        repo_id: selected.repo_id,
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

  const targetFolder = selected
    ? `${modelsRootDir}/${selected.folder_name}`.replace(/\\/g, "/")
    : "";

  return (
    <div
      className="dir-browser-backdrop"
      onClick={phase === "downloading" ? undefined : onClose}
      role="presentation"
    >
      <div
        className="dir-browser-panel hf-browse-panel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={pickOnly ? "Add quick preset from Hugging Face" : "Download LLM from Hugging Face"}
      >
        <div className="dir-browser-header">
          <h2>
            {phase === "browse" && (pickOnly ? "Add quick preset" : "Download LLM")}
            {phase === "confirm" && "Confirm download"}
            {phase === "downloading" && "Downloading model"}
            {phase === "complete" && "Download complete"}
            {phase === "failed" && "Download failed"}
          </h2>
          {phase !== "downloading" && (
            <button type="button" className="dir-browser-close" onClick={onClose} aria-label="Close">
              ×
            </button>
          )}
        </div>

        <div className="hf-browse-body">
          {phase === "browse" && (
            <>
              {!online && (
                <div className="hf-offline-banner">
                  No internet connection — Hugging Face browse and download are unavailable.
                </div>
              )}

              {recommendMode && hardware && (
                <div className="hf-hardware-banner">
                  <strong>Your system</strong>
                  <span>
                    Usable RAM: {hardware.usable_ram_gb} GB
                    {hardware.has_dedicated_gpu
                      ? ` · GPU: ${hardware.gpu_name} (${hardware.gpu_vram_gb} GB VRAM)`
                      : " · No dedicated CUDA GPU detected"}
                  </span>
                  <span className="hf-hardware-note">
                    Estimates use available RAM × 90% plus dedicated VRAM, with 1.2× model overhead.
                  </span>
                </div>
              )}

              <form className="hf-filter-grid" onSubmit={handleSearchSubmit}>
                <div className="hf-filter-field wide">
                  <label className="settings-label" htmlFor="hf-search">Search</label>
                  <input
                    id="hf-search"
                    className="settings-input"
                    type="search"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. coder security instruct"
                    disabled={!online}
                  />
                </div>
                <div className="hf-filter-field">
                  <label className="settings-label" htmlFor="hf-family">Model family</label>
                  <select
                    id="hf-family"
                    className="settings-input"
                    value={family}
                    onChange={(e) => setFamily(e.target.value)}
                    disabled={!online}
                  >
                    {filters.families.map((item) => (
                      <option key={item.id} value={item.id}>{item.label}</option>
                    ))}
                  </select>
                </div>
                <div className="hf-filter-field">
                  <label className="settings-label" htmlFor="hf-params">Parameters</label>
                  <select
                    id="hf-params"
                    className="settings-input"
                    value={paramSize}
                    onChange={(e) => setParamSize(e.target.value)}
                    disabled={!online}
                  >
                    {filters.param_sizes.map((item) => (
                      <option key={item.id} value={item.id}>{item.label}</option>
                    ))}
                  </select>
                </div>
                <div className="hf-filter-field">
                  <label className="settings-label" htmlFor="hf-purpose">Type</label>
                  <select
                    id="hf-purpose"
                    className="settings-input"
                    value={purpose}
                    onChange={(e) => setPurpose(e.target.value)}
                    disabled={!online}
                  >
                    {filters.purposes.map((item) => (
                      <option key={item.id} value={item.id}>{item.label}</option>
                    ))}
                  </select>
                </div>
                <div className="hf-filter-actions">
                  <button
                    type="button"
                    className={`settings-btn secondary compact${recommendMode ? " active" : ""}`}
                    onClick={handleRecommend}
                    disabled={!online || loading}
                  >
                    Recommend
                  </button>
                  <button
                    type="submit"
                    className="settings-btn primary compact"
                    disabled={!online || loading}
                  >
                    {loading ? "Searching…" : "Search"}
                  </button>
                </div>
              </form>

              {error && <div className="dir-browser-error">{error}</div>}

              <div className="hf-results">
                {!loading && online && models.length === 0 && (
                  <p className="hf-results-empty">No models match these filters. Try broader search terms.</p>
                )}
                <ul className="hf-model-list">
                  {models.map((model) => (
                    <li key={model.repo_id}>
                      <button
                        type="button"
                        className="hf-model-row"
                        onClick={() => handleSelectModel(model)}
                        disabled={!online}
                      >
                        <span className="hf-model-main">
                          <span className="hf-model-title-row">
                            <span className="hf-model-title">{model.label}</span>
                            {recommendMode && <FitBadge fit={model.fit} />}
                          </span>
                          <span className="hf-model-repo">{model.repo_id}</span>
                          {recommendMode && model.recommended_variant_id && (
                            <span className="hf-model-quant-hint">
                              Best start: {model.recommended_variant_id.replace("_", " ").toUpperCase()}
                            </span>
                          )}
                        </span>
                        <span className="hf-model-meta">
                          <span>{model.param_label}</span>
                          <span>{model.downloads.toLocaleString()} downloads</span>
                          {model.gated && <span className="hf-model-gated">Gated</span>}
                          {model.loader && !model.loader.compatible && (
                            <span className="hf-model-gated">Unsupported</span>
                          )}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
                {hasMore && online && (
                  <div className="hf-load-more">
                    <button
                      type="button"
                      className="settings-btn secondary compact"
                      onClick={() => runSearch(page + 1, true, recommendMode)}
                      disabled={loading}
                    >
                      Load more
                    </button>
                  </div>
                )}
              </div>
            </>
          )}

          {phase === "confirm" && selected && (
            <>
              <p className="model-download-lead">
                Download <strong>{selected.label}</strong> from Hugging Face?
              </p>
              <p className="model-download-copy">Repository: <code>{selected.repo_id}</code></p>
              <code className="model-download-path">{targetFolder}</code>

              <div className="settings-field hf-quant-field">
                <label className="settings-label" htmlFor="hf-quant-select">
                  Runtime quantization
                </label>
                <select
                  id="hf-quant-select"
                  className="settings-input"
                  value={selectedQuantId}
                  onChange={(e) => setSelectedQuantId(e.target.value)}
                >
                  {confirmVariants.map((variant) => (
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

              {selected.gated && (
                <p className="model-download-warning">
                  This model may require a Hugging Face account and license acceptance on huggingface.co.
                </p>
              )}
              {selected.loader && !selected.loader.compatible && (
                <p className="model-download-warning">
                  <strong>Not compatible with VULNERA:</strong> {selected.loader.note}
                </p>
              )}
              <p className="model-download-warning">
                Weights are downloaded once; Q4/Q8/FP16 controls how they are loaded at scan time.
              </p>
              {error && <div className="dir-browser-error">{error}</div>}
              <div className="model-download-actions">
                <button type="button" className="settings-btn secondary" onClick={() => setPhase("browse")}>
                  Back
                </button>
                <button
                  type="button"
                  className="settings-btn primary"
                  onClick={handleConfirmDownload}
                  disabled={selected.loader && !selected.loader.compatible}
                >
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
                {selected?.label} is in your models folder
                {selectedVariant ? ` with ${selectedVariant.label} load profile` : ""}.
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
