import { useCallback, useEffect, useMemo, useState } from "react";
import DirectoryBrowserModal from "../components/DirectoryBrowserModal";
import HuggingFaceBrowseModal from "../components/HuggingFaceBrowseModal";
import ModelDownloadModal from "../components/ModelDownloadModal";
import {
  deleteLlmModel,
  getConfig,
  getHfConnectivity,
  getLlmConfig,
  scanLlmModels,
  updateConfig,
  updateLlmConfig,
  updateQuickPresets,
} from "../api/client";
import "../styles/settings.css";

const PROVIDERS = [
  {
    id: "huggingface",
    label: "Local HuggingFace",
    description: "Run Qwen on this machine (GPU recommended). Falls back to mock if weights are missing.",
  },
  {
    id: "mock",
    label: "Mock (templates)",
    description: "Fast placeholder explanations — no GPU or model download required.",
  },
];

const SETTINGS_TABS = [
  { id: "pipeline", label: "Scan pipeline" },
  { id: "llm", label: "Local LLM model" },
];

function StatusBadge({ ok, label, detail }) {
  return (
    <span className={`settings-badge${ok ? " ok" : " warn"}`} title={detail}>
      {label}
    </span>
  );
}

function snapshotFromState(state) {
  return {
    llmProvider: state.llmProvider,
    maxFunctions: state.maxFunctions,
    useCpuEmbedder: state.useCpuEmbedder,
    llmMaxNewTokens: state.llmMaxNewTokens,
    llmMaxCodeChars: state.llmMaxCodeChars,
    llmExplanationEnabled: state.llmExplanationEnabled,
    llmChainOfThought: state.llmChainOfThought,
    modelsRootDir: state.modelsRootDir.trim(),
    selectedModelId: state.selectedModelId,
  };
}

function snapshotsEqual(a, b) {
  if (!a || !b) return false;
  return (
    a.llmProvider === b.llmProvider
    && a.maxFunctions === b.maxFunctions
    && a.useCpuEmbedder === b.useCpuEmbedder
    && a.llmMaxNewTokens === b.llmMaxNewTokens
    && a.llmMaxCodeChars === b.llmMaxCodeChars
    && a.llmExplanationEnabled === b.llmExplanationEnabled
    && a.llmChainOfThought === b.llmChainOfThought
    && a.modelsRootDir === b.modelsRootDir
    && a.selectedModelId === b.selectedModelId
  );
}

export default function SettingsPage() {
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("pipeline");
  const [savedSettings, setSavedSettings] = useState(null);

  const [llmProvider, setLlmProvider] = useState("huggingface");
  const [maxFunctions, setMaxFunctions] = useState(50);
  const [useCpuEmbedder, setUseCpuEmbedder] = useState(true);
  const [llmMaxNewTokens, setLlmMaxNewTokens] = useState(192);
  const [llmMaxCodeChars, setLlmMaxCodeChars] = useState(2400);
  const [llmExplanationEnabled, setLlmExplanationEnabled] = useState(true);
  const [llmChainOfThought, setLlmChainOfThought] = useState(true);

  const [modelsRootDir, setModelsRootDir] = useState("08_LLM/models");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [availableModels, setAvailableModels] = useState([]);
  const [modelStatus, setModelStatus] = useState(null);
  const [presets, setPresets] = useState([]);
  const [browseOpen, setBrowseOpen] = useState(false);
  const [modelsRootResolved, setModelsRootResolved] = useState("");
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [downloadPreset, setDownloadPreset] = useState(null);
  const [hfBrowseOpen, setHfBrowseOpen] = useState(false);
  const [hfPresetPickOpen, setHfPresetPickOpen] = useState(false);
  const [hfOnline, setHfOnline] = useState(true);
  const [editingPresets, setEditingPresets] = useState(false);
  const [presetBusy, setPresetBusy] = useState(false);
  const [maxQuickPresets, setMaxQuickPresets] = useState(3);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const currentSnapshot = useMemo(
    () => snapshotFromState({
      llmProvider,
      maxFunctions,
      useCpuEmbedder,
      llmMaxNewTokens,
      llmMaxCodeChars,
      llmExplanationEnabled,
      llmChainOfThought,
      modelsRootDir,
      selectedModelId,
    }),
    [
      llmProvider,
      maxFunctions,
      useCpuEmbedder,
      llmMaxNewTokens,
      llmMaxCodeChars,
      llmExplanationEnabled,
      llmChainOfThought,
      modelsRootDir,
      selectedModelId,
    ],
  );

  const isDirty = savedSettings !== null && !snapshotsEqual(currentSnapshot, savedSettings);

  const applySavedSnapshot = useCallback((snapshot) => {
    setLlmProvider(snapshot.llmProvider);
    setMaxFunctions(snapshot.maxFunctions);
    setUseCpuEmbedder(snapshot.useCpuEmbedder);
    setLlmMaxNewTokens(snapshot.llmMaxNewTokens);
    setLlmMaxCodeChars(snapshot.llmMaxCodeChars);
    setLlmExplanationEnabled(snapshot.llmExplanationEnabled);
    setLlmChainOfThought(snapshot.llmChainOfThought);
    setModelsRootDir(snapshot.modelsRootDir);
    setSelectedModelId(snapshot.selectedModelId);
  }, []);

  const applyScanResult = useCallback((scanResult, currentSelection = selectedModelId) => {
    const models = scanResult.available_models || [];
    setAvailableModels(models);
    setPresets(scanResult.presets || []);

    if (models.some((item) => item.id === currentSelection)) {
      setSelectedModelId(currentSelection);
    } else if (models.length) {
      setSelectedModelId(models[0].id);
    } else {
      setSelectedModelId("");
    }
  }, [selectedModelId]);

  const runScan = useCallback(async (rootDir, { keepSelection = true } = {}) => {
    const trimmed = rootDir.trim();
    if (!trimmed) {
      setAvailableModels([]);
      setSelectedModelId("");
      return;
    }

    setScanning(true);
    setError(null);
    try {
      const result = await scanLlmModels(trimmed);
      applyScanResult(result, keepSelection ? selectedModelId : "");
    } catch (err) {
      setError(err.message);
      setAvailableModels([]);
      setSelectedModelId("");
    } finally {
      setScanning(false);
    }
  }, [applyScanResult, selectedModelId]);

  const loadSettings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [webCfg, llmCfg] = await Promise.all([getConfig(), getLlmConfig()]);
      const scan = webCfg.scan || {};
      const model = llmCfg.model || {};
      const root = model.models_root_dir || "08_LLM/models";
      const selection = model.selected_model_id || "";
      const resolvedSelection = selection && (llmCfg.available_models || []).some((item) => item.id === selection)
        ? selection
        : (llmCfg.available_models || [])[0]?.id || "";

      const snapshot = snapshotFromState({
        llmProvider: scan.llm_provider || "mock",
        maxFunctions: scan.max_functions ?? 50,
        useCpuEmbedder: scan.use_cpu_embedder !== false,
        llmMaxNewTokens: scan.llm_max_new_tokens ?? 192,
        llmMaxCodeChars: scan.llm_max_code_chars ?? 2400,
        llmExplanationEnabled: scan.llm_explanation_enabled !== false,
        llmChainOfThought: scan.llm_chain_of_thought !== false,
        modelsRootDir: root,
        selectedModelId: resolvedSelection,
      });

      applySavedSnapshot(snapshot);
      setSavedSettings(snapshot);
      setModelsRootResolved(model.models_root_resolved || "");
      setModelStatus(model);
      setAvailableModels(llmCfg.available_models || []);
      setPresets(llmCfg.presets || []);
      setMaxQuickPresets(llmCfg.max_quick_presets ?? 3);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [applySavedSnapshot]);

  useEffect(() => {
    loadSettings();
    getHfConnectivity()
      .then((result) => setHfOnline(Boolean(result.online)))
      .catch(() => setHfOnline(false));
  }, [loadSettings]);

  const persistActiveModel = useCallback(async (folderName, label) => {
    const root = modelsRootDir.trim();
    if (!root || !folderName) return false;

    setPresetBusy(true);
    setError(null);
    try {
      const result = await updateLlmConfig({
        models_root_dir: root,
        selected_model_id: folderName,
      });
      setSelectedModelId(folderName);
      setModelStatus(result.model || null);
      setAvailableModels(result.available_models || []);
      setPresets(result.presets || []);
      setSavedSettings((prev) => (prev ? {
        ...prev,
        modelsRootDir: root,
        selectedModelId: folderName,
      } : prev));
      setMessage(`Active model: ${label || folderName}`);
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setPresetBusy(false);
    }
  }, [modelsRootDir]);

  const applyPreset = async (preset) => {
    const root = modelsRootDir.trim();
    setMessage(null);
    setError(null);

    if (editingPresets) return;

    if (preset.downloaded) {
      await persistActiveModel(preset.folder_name, preset.label);
      return;
    }

    if (!root) {
      setError("Select a models directory with Browse before downloading a model.");
      return;
    }

    setDownloadPreset(preset);
    setDownloadOpen(true);
  };

  const handleRemovePreset = async (presetId) => {
    const next = presets.filter((item) => item.id !== presetId);
    setPresetBusy(true);
    setError(null);
    try {
      const result = await updateQuickPresets(next.map((item) => ({
        id: item.id,
        repo_id: item.repo_id,
        label: item.label,
        folder_name: item.folder_name,
        note: item.note || "",
      })));
      setPresets(result.presets || []);
      setMessage("Quick preset removed.");
    } catch (err) {
      setError(err.message);
    } finally {
      setPresetBusy(false);
    }
  };

  const handleAddPresetPick = async (model) => {
    if (presets.length >= maxQuickPresets) {
      setError(`You can keep at most ${maxQuickPresets} quick presets.`);
      return;
    }
    if (presets.some((item) => item.repo_id === model.repo_id)) {
      setError("That model is already in your quick presets.");
      return;
    }

    const entry = {
      repo_id: model.repo_id,
      label: model.label,
      folder_name: model.folder_name,
      note: model.param_label ? `~${model.param_label} parameters` : "",
    };

    setPresetBusy(true);
    setError(null);
    try {
      const result = await updateQuickPresets([
        ...presets.map((item) => ({
          id: item.id,
          repo_id: item.repo_id,
          label: item.label,
          folder_name: item.folder_name,
          note: item.note || "",
        })),
        entry,
      ]);
      setPresets(result.presets || []);
      setHfPresetPickOpen(false);
      setMessage(`Added ${model.label} to quick presets.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setPresetBusy(false);
    }
  };

  const handleHfDownloadComplete = async (model) => {
    setHfBrowseOpen(false);
    await runScan(modelsRootDir, { keepSelection: false });
    try {
      await updateLlmConfig({
        models_root_dir: modelsRootDir.trim(),
        selected_model_id: model.folder_name,
        quantization_id: model.quantization_id || "q4_nf4",
      });
      await loadSettings();
      setMessage(`${model.label} downloaded with ${(model.quantization_id || "q4_nf4").replace("_", " ").toUpperCase()} profile.`);
    } catch {
      setSelectedModelId(model.folder_name);
      setMessage(`${model.label} downloaded and selected. Save settings to keep the quantization profile.`);
    }
  };

  const handleDownloadComplete = async (item) => {
    setDownloadOpen(false);
    setDownloadPreset(null);
    await runScan(modelsRootDir, { keepSelection: false });
    try {
      await updateLlmConfig({
        models_root_dir: modelsRootDir.trim(),
        selected_model_id: item.folder_name,
        quantization_id: item.quantization_id || "q4_nf4",
      });
      await loadSettings();
      setMessage(`${item.label || item.folder_name} downloaded with ${(item.quantization_id || "q4_nf4").replace("_", " ").toUpperCase()} profile.`);
    } catch {
      setSelectedModelId(item.folder_name);
      setMessage(`${item.label || item.folder_name} downloaded and selected. Save settings to keep it.`);
    }
  };

  const openHfBrowse = () => {
    if (!modelsRootDir.trim()) {
      setError("Select a models directory with Browse before downloading a model.");
      return;
    }
    setError(null);
    setHfBrowseOpen(true);
  };

  const handleDeleteModel = async () => {
    if (!deleteTarget || !modelsRootDir.trim()) return;

    setDeleting(true);
    setError(null);
    setMessage(null);
    try {
      const result = await deleteLlmModel({
        models_root_dir: modelsRootDir.trim(),
        model_id: deleteTarget.id,
      });
      const nextModels = result.available_models || [];
      const nextSelected = result.model?.selected_model_id || nextModels[0]?.id || "";

      setAvailableModels(nextModels);
      setPresets(result.presets || []);
      setModelStatus(result.model || null);
      setSelectedModelId(nextSelected);
      setSavedSettings((prev) => (prev ? {
        ...prev,
        modelsRootDir: modelsRootDir.trim(),
        selectedModelId: nextSelected,
      } : prev));

      const switched = deleteTarget.id !== nextSelected && nextSelected;
      setMessage(
        switched
          ? `Deleted ${deleteTarget.label}. Active model switched to ${nextModels.find((m) => m.id === nextSelected)?.label || nextSelected}.`
          : `Deleted ${deleteTarget.label} from disk.`,
      );
      setDeleteTarget(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setDeleting(false);
    }
  };

  const handleDirectorySelect = async ({ storage_path, current_path }) => {
    setModelsRootDir(storage_path);
    setModelsRootResolved(current_path || "");
    setMessage(null);
    await runScan(storage_path, { keepSelection: false });
  };

  const handleScanClick = () => {
    runScan(modelsRootDir, { keepSelection: false });
  };

  const handleReset = () => {
    if (!savedSettings) return;
    setMessage(null);
    setError(null);
    applySavedSnapshot(savedSettings);
  };

  const handleSave = async () => {
    const llmModelDirty = savedSettings && (
      currentSnapshot.modelsRootDir !== savedSettings.modelsRootDir
      || currentSnapshot.selectedModelId !== savedSettings.selectedModelId
    );
    const webConfigDirty = savedSettings && (
      currentSnapshot.llmProvider !== savedSettings.llmProvider
      || currentSnapshot.maxFunctions !== savedSettings.maxFunctions
      || currentSnapshot.useCpuEmbedder !== savedSettings.useCpuEmbedder
      || currentSnapshot.llmMaxNewTokens !== savedSettings.llmMaxNewTokens
      || currentSnapshot.llmMaxCodeChars !== savedSettings.llmMaxCodeChars
      || currentSnapshot.llmExplanationEnabled !== savedSettings.llmExplanationEnabled
      || currentSnapshot.llmChainOfThought !== savedSettings.llmChainOfThought
    );

    if (llmModelDirty && !selectedModelId) {
      setError("Scan your models directory and select a downloaded model first.");
      return;
    }

    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      const tasks = [];
      if (webConfigDirty) {
        tasks.push({
          key: "web",
          run: () => updateConfig({
            llm_provider: llmProvider,
            max_functions: maxFunctions,
            use_cpu_embedder: useCpuEmbedder,
            llm_max_new_tokens: llmMaxNewTokens,
            llm_max_code_chars: llmMaxCodeChars,
            llm_explanation_enabled: llmExplanationEnabled,
            llm_chain_of_thought: llmChainOfThought,
          }),
        });
      }
      if (llmModelDirty) {
        tasks.push({
          key: "llm",
          run: () => updateLlmConfig({
            models_root_dir: modelsRootDir.trim(),
            selected_model_id: selectedModelId,
          }),
        });
      }

      const results = await Promise.all(tasks.map((task) => task.run()));
      const resultByKey = Object.fromEntries(tasks.map((task, index) => [task.key, results[index]]));

      const webResult = resultByKey.web;
      const llmResult = resultByKey.llm;

      let nextLlmProvider = llmProvider;
      let nextSelectedModelId = selectedModelId;
      let saveMessage = "Settings saved. New scans will use these values.";

      if (webResult?.scan) {
        nextLlmProvider = webResult.scan.llm_provider || llmProvider;
        setLlmProvider(nextLlmProvider);
      }

      if (llmResult) {
        const hadLoaded = modelStatus?.cached_in_memory;
        const oldPath = modelStatus?.loaded_model_path || modelStatus?.resolved_path || "";
        const newPath = llmResult.model?.resolved_path || "";
        nextSelectedModelId = llmResult.model?.selected_model_id || selectedModelId;
        setModelStatus(llmResult.model || null);
        setAvailableModels(llmResult.available_models || []);
        setPresets(llmResult.presets || []);
        setSelectedModelId(nextSelectedModelId);
        if (hadLoaded && oldPath && newPath && oldPath !== newPath) {
          saveMessage = "Settings saved. The running LLM stays loaded until your next scan switches to the new selection.";
        }
      }

      setMessage(saveMessage);
      setSavedSettings(snapshotFromState({
        llmProvider: nextLlmProvider,
        maxFunctions,
        useCpuEmbedder,
        llmMaxNewTokens,
        llmMaxCodeChars,
        llmExplanationEnabled,
        llmChainOfThought,
        modelsRootDir,
        selectedModelId: nextSelectedModelId,
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="settings-workspace">
        <div className="settings-loading">Loading settings…</div>
      </div>
    );
  }

  const selectedModel = availableModels.find((item) => item.id === selectedModelId);

  return (
    <div className="settings-workspace">
      <div className="settings-chrome">
        <div className="settings-titlebar">
          <span className="settings-title">VULNERA — Settings</span>
        </div>
      </div>

      <div className="settings-body">
        <header className="settings-intro">
          <h1>Configuration</h1>
          <p>
            Pipeline and scan limits are saved with Save settings. Model selection, presets, and downloads apply immediately.
          </p>
        </header>

        <nav className="settings-tabs" aria-label="Settings categories">
          {SETTINGS_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`settings-tab${activeTab === tab.id ? " active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              aria-current={activeTab === tab.id ? "page" : undefined}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {message && <div className="settings-banner ok">{message}</div>}
        {error && <div className="settings-banner error">{error}</div>}

        {activeTab === "pipeline" && (
          <section className="settings-panel">
            <div className="settings-panel-header">
              <span className="settings-panel-title">Scan pipeline</span>
              <span className="settings-panel-sub">web_config.yaml</span>
            </div>
            <div className="settings-panel-body">
              <div className="settings-field">
                <label className="settings-label" htmlFor="llm-provider">
                  Explanation provider
                </label>
                <select
                  id="llm-provider"
                  className="settings-input"
                  value={llmProvider}
                  onChange={(e) => setLlmProvider(e.target.value)}
                >
                  {PROVIDERS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <p className="settings-hint">
                  {PROVIDERS.find((option) => option.id === llmProvider)?.description}
                </p>
              </div>

              <div className="settings-field">
                <label className="settings-label" htmlFor="max-functions">
                  Max functions per scan
                </label>
                <input
                  id="max-functions"
                  className="settings-input"
                  type="number"
                  min={1}
                  max={500}
                  value={maxFunctions}
                  onChange={(e) => setMaxFunctions(Number(e.target.value))}
                />
                <p className="settings-hint">
                  Caps how many functions are scored and explained in a single scan job.
                </p>
              </div>

              <div className="settings-field settings-field-check">
                <label className="settings-check">
                  <input
                    type="checkbox"
                    checked={useCpuEmbedder}
                    onChange={(e) => setUseCpuEmbedder(e.target.checked)}
                  />
                  <span>Use CPU for GraphCodeBERT embedder</span>
                </label>
                <p className="settings-hint">
                  GraphCodeBERT base is fixed to match the trained ensemble. This toggle only
                  controls CPU vs GPU for embedding. Keeps GPU memory free for the local LLM on 8GB cards.
                </p>
              </div>
            </div>
          </section>
        )}

        {activeTab === "llm" && (
          <section className="settings-panel">
            <div className="settings-panel-header">
              <span className="settings-panel-title">Local LLM model</span>
              <span className="settings-panel-sub">llm_config.yaml</span>
            </div>
            <div className="settings-panel-body">
              <div className="settings-field settings-field-check settings-llm-master">
                <label className="settings-check">
                  <input
                    type="checkbox"
                    checked={llmExplanationEnabled}
                    onChange={(e) => setLlmExplanationEnabled(e.target.checked)}
                  />
                  <span>Enable LLM explanations</span>
                </label>
                <p className="settings-hint">
                  Generate natural-language explanations during scans. Turn off for ML-only
                  triage — faster scans and no Qwen GPU load.
                </p>
              </div>

              <div className="settings-field settings-field-check">
                <label className="settings-check">
                  <input
                    type="checkbox"
                    checked={llmChainOfThought}
                    onChange={(e) => setLlmChainOfThought(e.target.checked)}
                    disabled={!llmExplanationEnabled}
                  />
                  <span>Thinking mode (chain-of-thought)</span>
                </label>
                <p className="settings-hint">
                  Adds step-by-step reasoning to explanation prompts. Disable for faster
                  explanation generation when explanations are enabled.
                </p>
              </div>

              {modelStatus && (
                <div className="settings-status-row">
                  <StatusBadge
                    ok={modelStatus.downloaded}
                    label={modelStatus.downloaded ? "Selected model on disk" : "No model selected"}
                    detail={modelStatus.resolved_path}
                  />
                  <StatusBadge
                    ok={modelStatus.cached_in_memory}
                    label={modelStatus.cached_in_memory ? "Model loaded in memory" : "Not loaded yet"}
                    detail={modelStatus.loaded_model_path || "Loads on next scan"}
                  />
                </div>
              )}

              <div className="settings-field">
                <div className="settings-presets-header">
                  <span className="settings-label">Quick presets</span>
                  <button
                    type="button"
                    className={`settings-btn secondary compact${editingPresets ? " active" : ""}`}
                    onClick={() => {
                      setEditingPresets((value) => !value);
                      setMessage(null);
                      setError(null);
                    }}
                    disabled={presetBusy}
                  >
                    {editingPresets ? "Done" : "Edit"}
                  </button>
                </div>
                <p className="settings-hint">
                  {editingPresets
                    ? `Keep up to ${maxQuickPresets} models. Click Done when finished editing.`
                    : "Click a downloaded preset to switch the active model immediately."}
                </p>
                <div className="settings-presets">
                  {presets.map((preset) => {
                    const isActive = selectedModelId === preset.folder_name;
                    return (
                      <div key={preset.id} className="settings-preset-slot">
                        <button
                          type="button"
                          className={`settings-preset-btn${preset.downloaded ? " ready" : " missing"}${isActive ? " active" : ""}`}
                          onClick={() => applyPreset(preset)}
                          title={preset.note}
                          disabled={presetBusy || editingPresets}
                        >
                          <span className="settings-preset-row">
                            <span className="settings-preset-label">{preset.label}</span>
                            <span className={`settings-preset-status${preset.downloaded ? " ok" : ""}${isActive ? " active" : ""}`}>
                              {isActive ? "Active" : preset.downloaded ? "Downloaded" : "Not downloaded"}
                            </span>
                          </span>
                          <span className="settings-preset-note">{preset.note || preset.repo_id}</span>
                        </button>
                        {editingPresets && (
                          <button
                            type="button"
                            className="settings-preset-remove"
                            onClick={() => handleRemovePreset(preset.id)}
                            disabled={presetBusy}
                            aria-label={`Remove ${preset.label}`}
                            title="Remove preset"
                          >
                            ×
                          </button>
                        )}
                      </div>
                    );
                  })}
                  {editingPresets && presets.length < maxQuickPresets && (
                    <button
                      type="button"
                      className="settings-preset-btn settings-preset-add"
                      onClick={() => {
                        setError(null);
                        setHfPresetPickOpen(true);
                      }}
                      disabled={presetBusy || !hfOnline}
                    >
                      <span className="settings-preset-label">+ Add model</span>
                      <span className="settings-preset-note">
                        {hfOnline
                          ? "Browse Hugging Face to add a shortcut"
                          : "No internet — cannot browse Hugging Face"}
                      </span>
                    </button>
                  )}
                  {!editingPresets && presets.length === 0 && (
                    <p className="settings-hint">No quick presets yet — click Edit to add up to {maxQuickPresets}.</p>
                  )}
                </div>
              </div>

              <div className="settings-field">
                <span className="settings-label">Download LLM</span>
                <button
                  type="button"
                  className={`settings-preset-btn hf-download-entry${hfOnline ? "" : " disabled"}`}
                  onClick={openHfBrowse}
                  disabled={!hfOnline}
                  title={
                    hfOnline
                      ? "Browse and download models from Hugging Face"
                      : "No internet connection"
                  }
                >
                  <span className="settings-preset-row">
                    <span className="settings-preset-label">Browse Hugging Face</span>
                    <span className={`settings-preset-status${hfOnline ? " ok" : ""}`}>
                      {hfOnline ? "Online" : "No internet connection"}
                    </span>
                  </span>
                  <span className="settings-preset-note">
                    Search by family, parameter size, and download directly into your models folder.
                  </span>
                </button>
              </div>

              <div className="settings-field">
                <span className="settings-label">Models directory</span>
                <div className="settings-path-row">
                  <div
                    className={`settings-path-display mono${modelsRootDir ? "" : " empty"}`}
                    title={modelsRootResolved || modelsRootDir || "No directory selected"}
                  >
                    {modelsRootResolved || modelsRootDir || "No directory selected — use Browse"}
                  </div>
                  <button
                    type="button"
                    className="settings-btn secondary compact"
                    onClick={() => setBrowseOpen(true)}
                  >
                    Browse…
                  </button>
                  <button
                    type="button"
                    className="settings-btn secondary compact"
                    onClick={handleScanClick}
                    disabled={scanning || !modelsRootDir.trim()}
                  >
                    {scanning ? "Scanning…" : "Scan"}
                  </button>
                </div>
                <p className="settings-hint">
                  Folder containing one subdirectory per downloaded HuggingFace model.
                </p>
              </div>

              <div className="settings-field">
                <label className="settings-label" htmlFor="model-select">
                  Active model
                </label>
                <div className="settings-model-row">
                  <select
                    id="model-select"
                    className="settings-input"
                    value={selectedModelId}
                    onChange={async (e) => {
                      const nextId = e.target.value;
                      const model = availableModels.find((item) => item.id === nextId);
                      if (!nextId) {
                        setSelectedModelId("");
                        return;
                      }
                      setSelectedModelId(nextId);
                      await persistActiveModel(nextId, model?.label);
                    }}
                    disabled={!availableModels.length || presetBusy || deleting}
                  >
                    {!availableModels.length ? (
                      <option value="">No downloaded models — pick a preset above to download one</option>
                    ) : (
                      availableModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))
                    )}
                  </select>
                  {selectedModel && (
                    <button
                      type="button"
                      className="settings-btn danger compact"
                      onClick={() => {
                        setError(null);
                        setDeleteTarget(selectedModel);
                      }}
                      disabled={presetBusy || deleting || !modelsRootDir.trim()}
                      title={`Delete ${selectedModel.label} from disk`}
                    >
                      Delete
                    </button>
                  )}
                </div>
                {selectedModel && (
                  <p className="settings-hint">
                    Folder: <code>{selectedModel.relative_path}</code>
                    {" · "}
                    Permanently removes weights from your models directory.
                  </p>
                )}
              </div>

              <div className="settings-grid two">
                <div className="settings-field">
                  <label className="settings-label" htmlFor="max-tokens">
                    LLM max new tokens
                  </label>
                  <input
                    id="max-tokens"
                    className="settings-input"
                    type="number"
                    min={32}
                    max={2048}
                    value={llmMaxNewTokens}
                    onChange={(e) => setLlmMaxNewTokens(Number(e.target.value))}
                  />
                </div>
                <div className="settings-field">
                  <label className="settings-label" htmlFor="max-code-chars">
                    Max code chars per window
                  </label>
                  <input
                    id="max-code-chars"
                    className="settings-input"
                    type="number"
                    min={256}
                    max={16000}
                    value={llmMaxCodeChars}
                    onChange={(e) => setLlmMaxCodeChars(Number(e.target.value))}
                  />
                </div>
              </div>
            </div>
          </section>
        )}

        <footer className={`settings-actions${isDirty ? " dirty" : ""}`}>
          {!isDirty && (
            <span className="settings-actions-status">All changes saved</span>
          )}
          <button
            type="button"
            className="settings-btn secondary"
            onClick={handleReset}
            disabled={saving || !isDirty}
          >
            Reset
          </button>
          <button
            type="button"
            className="settings-btn primary"
            onClick={handleSave}
            disabled={saving || !isDirty}
          >
            {saving ? "Saving…" : "Save settings"}
          </button>
        </footer>
      </div>

      <DirectoryBrowserModal
        open={browseOpen}
        initialPath={modelsRootDir}
        onClose={() => setBrowseOpen(false)}
        onSelect={handleDirectorySelect}
      />

      <ModelDownloadModal
        open={downloadOpen}
        preset={downloadPreset}
        modelsRootDir={modelsRootDir}
        onClose={() => {
          setDownloadOpen(false);
          setDownloadPreset(null);
        }}
        onComplete={handleDownloadComplete}
      />

      <HuggingFaceBrowseModal
        open={hfBrowseOpen}
        modelsRootDir={modelsRootDir}
        onClose={() => setHfBrowseOpen(false)}
        onComplete={handleHfDownloadComplete}
      />

      <HuggingFaceBrowseModal
        open={hfPresetPickOpen}
        pickOnly
        onClose={() => setHfPresetPickOpen(false)}
        onPick={handleAddPresetPick}
      />

      {deleteTarget && (
        <div
          className="dir-browser-backdrop"
          onClick={deleting ? undefined : () => setDeleteTarget(null)}
          role="presentation"
        >
          <div
            className="dir-browser-panel settings-delete-panel"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label={`Delete ${deleteTarget.label}`}
          >
            <div className="dir-browser-header">
              <h2>Delete model</h2>
              {!deleting && (
                <button
                  type="button"
                  className="dir-browser-close"
                  onClick={() => setDeleteTarget(null)}
                  aria-label="Close"
                >
                  ×
                </button>
              )}
            </div>
            <div className="settings-delete-body">
              <p className="model-download-lead">
                Permanently delete <strong>{deleteTarget.label}</strong>?
              </p>
              <p className="settings-hint">
                This removes <code>{deleteTarget.relative_path}</code> and all weight files on disk.
                {selectedModelId === deleteTarget.id && availableModels.length > 1
                  ? " Another downloaded model will become active automatically."
                  : ""}
                {modelStatus?.cached_in_memory && selectedModelId === deleteTarget.id
                  ? " The model will be unloaded from GPU memory first."
                  : ""}
              </p>
              <div className="settings-delete-actions">
                <button
                  type="button"
                  className="settings-btn secondary"
                  onClick={() => setDeleteTarget(null)}
                  disabled={deleting}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="settings-btn danger"
                  onClick={handleDeleteModel}
                  disabled={deleting}
                >
                  {deleting ? "Deleting…" : "Delete model"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
