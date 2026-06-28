const API_BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  if (response.status === 204) return null;
  return response.json();
}

export async function getHealth() {
  return request("/api/health");
}

export async function getConfig() {
  return request("/api/config");
}

export async function updateConfig(scan) {
  return request("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scan }),
  });
}

export async function getLlmConfig() {
  return request("/api/llm-config");
}

export async function scanLlmModels(modelsRootDir) {
  const params = new URLSearchParams({ models_root_dir: modelsRootDir });
  return request(`/api/llm-config/scan?${params}`);
}

export async function browseDirectory(path) {
  const query = path ? `?${new URLSearchParams({ path })}` : "";
  return request(`/api/llm-config/browse${query}`);
}

export async function startModelDownload({ models_root_dir, preset_id, repo_id, quantization_id }) {
  return request("/api/llm-config/downloads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ models_root_dir, preset_id, repo_id, quantization_id }),
  });
}

export async function getHfConnectivity() {
  return request("/api/llm-config/connectivity");
}

export async function getHfFilters() {
  return request("/api/llm-config/hf-filters");
}

export async function searchHfModels({
  query = "",
  family = "all",
  param_size = "any",
  purpose = "code",
  page = 0,
  include_fit = false,
}) {
  const params = new URLSearchParams({
    query,
    family,
    param_size,
    purpose,
    page: String(page),
  });
  if (include_fit) {
    params.set("include_fit", "true");
  }
  return request(`/api/llm-config/hf-models?${params}`);
}

export async function getLlmHardware() {
  return request("/api/llm-config/hardware");
}

export async function getHfModelVariants(repoId) {
  const params = new URLSearchParams({ repo_id: repoId });
  return request(`/api/llm-config/hf-models/variants?${params}`);
}

export async function getHfRuntimeQuants() {
  return request("/api/llm-config/hf-quants");
}

export async function getModelDownloadJob(jobId) {
  return request(`/api/llm-config/downloads/${jobId}`);
}

export function streamModelDownload(jobId, onEvent) {
  const source = new EventSource(`${API_BASE}/api/llm-config/downloads/${jobId}/stream`);
  let pollTimer = null;
  let closed = false;

  const finish = () => {
    if (closed) return;
    closed = true;
    source.close();
    if (pollTimer) clearInterval(pollTimer);
  };

  const handleSnapshot = (data) => {
    onEvent(data);
    if (data.status === "completed" || data.status === "failed") {
      finish();
    }
  };

  source.onmessage = (event) => {
    handleSnapshot(JSON.parse(event.data));
  };

  source.onerror = () => {
    source.close();
    if (closed || pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const data = await getModelDownloadJob(jobId);
        handleSnapshot(data);
      } catch {
        finish();
      }
    }, 1000);
  };

  return { close: finish };
}

export async function updateLlmConfig(model) {
  return request("/api/llm-config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(model),
  });
}

export async function updateQuickPresets(quickPresets) {
  return request("/api/llm-config/quick-presets", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quick_presets: quickPresets }),
  });
}

export async function deleteLlmModel({ models_root_dir, model_id }) {
  return request("/api/llm-config/models", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ models_root_dir, model_id }),
  });
}

export async function listScans() {
  return request("/api/scans");
}

export async function listStandaloneScans() {
  return request("/api/scans?standalone=true");
}

export async function listProjects() {
  return request("/api/projects");
}

export async function createProject(name) {
  return request("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function getProject(projectId) {
  return request(`/api/projects/${projectId}`);
}

export async function deleteProject(projectId) {
  return request(`/api/projects/${projectId}`, { method: "DELETE" });
}

export async function getScan(scanId) {
  return request(`/api/scans/${scanId}`);
}

function filenameFromDisposition(header, fallback) {
  if (!header) return fallback;
  const match = /filename\*?=(?:UTF-8''|")?([^";]+)/i.exec(header);
  if (!match) return fallback;
  try {
    return decodeURIComponent(match[1].replace(/"/g, ""));
  } catch {
    return match[1].replace(/"/g, "") || fallback;
  }
}

export async function exportScanReport(scanId, { filename } = {}) {
  const response = await fetch(`${API_BASE}/api/scans/${scanId}/report.pdf`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Report export failed: ${response.status}`);
  }
  const blob = await response.blob();
  const fallback = filename || `vulnera_report_${scanId}.pdf`;
  const downloadName = filenameFromDisposition(
    response.headers.get("Content-Disposition"),
    fallback,
  );
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  return downloadName;
}

export async function deleteScan(scanId) {
  return request(`/api/scans/${scanId}`, { method: "DELETE" });
}

export async function startScan(file, { llmProvider, maxFunctions, projectId } = {}) {
  const form = new FormData();
  form.append("file", file);
  if (llmProvider) form.append("llm_provider", llmProvider);
  if (maxFunctions != null) form.append("max_functions", String(maxFunctions));
  if (projectId) form.append("project_id", projectId);
  return request("/api/scans", { method: "POST", body: form });
}

export function waitForJob(jobId) {
  return new Promise((resolve, reject) => {
    const connection = streamJob(jobId, (snapshot) => {
      if (snapshot.status === "completed") {
        connection.close();
        resolve(snapshot);
      }
      if (snapshot.status === "failed") {
        connection.close();
        reject(new Error(snapshot.error || "Scan failed"));
      }
    });
  });
}

export async function getJob(jobId) {
  return request(`/api/scans/jobs/${jobId}`);
}

export function streamJob(jobId, onEvent) {
  const source = new EventSource(`${API_BASE}/api/scans/jobs/${jobId}/stream`);
  let pollTimer = null;
  let closed = false;

  const finish = () => {
    if (closed) return;
    closed = true;
    source.close();
    if (pollTimer) clearInterval(pollTimer);
  };

  const handleSnapshot = (data) => {
    onEvent(data);
    if (data.status === "completed" || data.status === "failed") {
      finish();
    }
  };

  source.onmessage = (event) => {
    handleSnapshot(JSON.parse(event.data));
  };

  source.onerror = () => {
    source.close();
    if (closed || pollTimer) return;
    // SSE can drop when backend reloads, loads LLM, or proxy resets — poll as fallback.
    pollTimer = setInterval(async () => {
      try {
        const data = await getJob(jobId);
        handleSnapshot(data);
      } catch {
        finish();
      }
    }, 1000);
  };

  return { close: finish };
}
