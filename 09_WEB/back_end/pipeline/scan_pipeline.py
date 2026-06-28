"""End-to-end scan orchestration for uploaded C/C++ files."""

from __future__ import annotations

import copy
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.signature_runtime import attach_signature_attribution
from pipeline.file_extractor import ExtractedFunction, extract_functions_from_source
from pipeline.model_runtime import (
    encode_scan_windows,
    get_encoder,
    get_model_bundle,
    get_pipeline_helpers,
    release_encoder,
    release_gpu_for_llm,
    release_model_bundle,
    score_function_windows,
)
from pipeline.progress_tracker import ProgressCallback, ScanProgressTracker
from pipeline.file_score_bridge import build_file_score

PHASES: list[tuple[str, str]] = [
    ("extract", "Extract"),
    ("clean", "Clean"),
    ("window", "Window"),
    ("embed", "Embed"),
    ("trees", "Trees"),
    ("calibrate", "Calibrate"),
    ("window_detect", "Window detect"),
    ("aggregation", "Aggregation"),
    ("signature_match", "Signature match"),
    ("explain", "Explain"),
]

STATUS_HIGHLIGHT = {
    "agree_positive": "vuln",
    "review_suggested": "review",
    "diffuse_risk": "diffuse",
    "agree_negative": "safe",
}


@dataclass
class ScanResult:
    scan_id: str
    filename: str
    uploaded_at: str
    source_code: str
    functions: list[dict[str, Any]]
    file_markers: list[dict[str, Any]]
    file_score: dict[str, Any] = field(default_factory=dict)
    phase_log: list[dict[str, Any]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    llm_explanation_enabled: bool = True
    llm_chain_of_thought: bool = True


def _log_gpu_memory(label: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(
            f"GPU memory ({label}): {used / 1e9:.2f} GB used / {total / 1e9:.2f} GB total",
            flush=True,
        )
    except Exception:
        pass


def _release_token_counter(counter: Any) -> None:
    tokenizer = getattr(counter, "_tokenizer", None)
    if tokenizer is not None:
        del counter._tokenizer
        counter._tokenizer = None


def _count_explain_steps(
    functions: list[dict[str, Any]],
    *,
    use_llm: bool,
    explanations_enabled: bool = True,
) -> int:
    if not explanations_enabled:
        return 0
    if not use_llm:
        return len(functions)
    total = 1  # local LLM load
    for function in functions:
        markers = function.get("markers") or []
        status = str(function.get("agreement_status", "agree_negative"))
        actionable = [m for m in markers if str(m.get("highlight_kind")) != "safe"]
        if status == "agree_negative" or not actionable:
            total += 1
        else:
            total += len(actionable)
    return total


def _relaxed_clean_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    relaxed = copy.deepcopy(cfg)
    relaxed.setdefault("3_cleaner", {})
    relaxed["3_cleaner"].update(
        {
            "keep_comments": False,
            "normalize_whitespace": True,
            "drop_empty_code": True,
            "min_lines": 1,
            "min_tokens": 5,
        }
    )
    return relaxed


def _locate_window_lines(full_code: str, window_code: str) -> tuple[int, int]:
    full_lines = full_code.splitlines()
    window_stripped = window_code.strip()
    if not window_stripped:
        return 1, max(1, len(full_lines))

    position = full_code.find(window_stripped)
    if position >= 0:
        start = full_code[:position].count("\n") + 1
        end = start + window_stripped.count("\n")
        return start, max(start, end)

    window_lines = [line for line in window_stripped.splitlines() if line.strip()]
    first = window_lines[0].strip()
    for index, line in enumerate(full_lines):
        if line.strip() == first or first in line:
            start = index + 1
            end = min(start + len(window_lines) - 1, len(full_lines))
            return start, end
    return 1, len(full_lines)


def _offset_markers_to_file(markers: list[dict[str, Any]], file_start_line: int) -> list[dict[str, Any]]:
    offset = max(0, int(file_start_line) - 1)
    shifted: list[dict[str, Any]] = []
    for marker in markers:
        item = dict(marker)
        item["line"] = int(marker["line"]) + offset
        item["end_line"] = int(marker.get("end_line", marker["line"])) + offset
        shifted.append(item)
    return shifted


def _map_cleaned_lines(line_map: list[int], local_start: int, local_end: int) -> tuple[int, int]:
    if not line_map:
        return local_start, local_end
    start_index = max(0, min(local_start - 1, len(line_map) - 1))
    end_index = max(0, min(local_end - 1, len(line_map) - 1))
    if end_index < start_index:
        end_index = start_index
    return line_map[start_index], line_map[end_index]


def merge_file_markers(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for function in functions:
        status = str(function.get("agreement_status", "agree_negative"))
        alert_line = int(function.get("code_start_line") or function.get("file_start_line") or 1)
        display = function.get("status_display") or {}
        merged.append(
            {
                "line": alert_line,
                "end_line": alert_line,
                "marker_type": "function_alert",
                "status": status,
                "highlight_kind": STATUS_HIGHLIGHT.get(status, "safe"),
                "window_index": None,
                "window_prob": float(function.get("max_window_prob") or function.get("function_score_calibrated") or 0.0),
                "title": f"{function.get('name', 'function')} · {display.get('label', status)}",
                "explanation": function.get("explanation", ""),
                "function_score_calibrated": float(function.get("function_score_calibrated", 0.0)),
                "max_window_prob": float(function.get("max_window_prob") or 0.0),
                "function_id": function.get("function_group_id"),
                "function_name": function.get("name"),
            }
        )
        for marker in function.get("markers") or []:
            merged.append(
                {
                    **marker,
                    "marker_type": marker.get("marker_type", "window"),
                    "function_score_calibrated": float(function.get("function_score_calibrated", 0.0)),
                    "function_id": function.get("function_group_id"),
                    "function_name": function.get("name"),
                }
            )
    merged.sort(key=lambda item: (int(item["line"]), 0 if item.get("marker_type") == "function_alert" else 1))
    return merged


def _window_marker_meta(
    function: dict[str, Any],
    *,
    window_index: int,
) -> tuple[str, str]:
    """Return (highlight_kind, label) for one window row."""
    status = str(function.get("agreement_status", "agree_negative"))
    flagged = {int(index) for index in function.get("flagged_window_indices") or []}
    contributing = {int(index) for index in function.get("contributing_window_indices") or []}

    if window_index in flagged:
        if status == "agree_positive":
            return "vuln", "Vulnerable"
        if status == "review_suggested":
            return "review", "Review suggested"
        return "review", "Flagged window"
    if window_index in contributing and status == "diffuse_risk":
        return "diffuse", "Diffuse contributor"
    return "safe", "Safe"


def _build_markers(
    function: dict[str, Any],
    function_code: str,
    *,
    line_map: list[int],
    file_end_line: int,
) -> list[dict[str, Any]]:
    status = str(function["agreement_status"])
    prompt_windows = function.get("prompt_windows") or []
    markers: list[dict[str, Any]] = []
    for window in prompt_windows:
        window_index = int(window["window_index"])
        highlight_kind, label = _window_marker_meta(function, window_index=window_index)
        local_start, local_end = _locate_window_lines(function_code, str(window.get("code", "")))
        start_line, end_line = _map_cleaned_lines(line_map, local_start, local_end)
        end_line = min(file_end_line, end_line)
        markers.append(
            {
                "line": start_line,
                "end_line": end_line,
                "window_index": window_index,
                "status": status,
                "highlight_kind": highlight_kind,
                "window_prob": float(window.get("window_prob", 0.0)),
                "title": f"Window {window_index} · {label}",
                "explanation": "",
            }
        )
    return markers


def _mock_function_explanation(function: dict[str, Any]) -> str:
    status = str(function["agreement_status"])
    score = float(function.get("function_score_calibrated", 0.0))
    name = function.get("name", "function")
    if status == "agree_positive":
        return (
            f"Function `{name}` is flagged as vulnerable (calibrated risk {score:.1%}). "
            "Both the function-level ensemble and at least one window-level detector agree. "
            "Review memory safety, bounds checks, and untrusted input handling across the whole function."
        )
    if status == "review_suggested":
        return (
            f"Function `{name}` needs review: window-level signal crossed threshold while pooled "
            f"function risk stayed at {score:.1%}. Localized unsafe logic may be diluted by max-pooling."
        )
    if status == "diffuse_risk":
        return (
            f"Function `{name}` is flagged with diffuse cross-window risk (calibrated {score:.1%}). "
            "No single window crossed threshold, but pooled contributors elevate overall risk."
        )
    return (
        f"Function `{name}` is not flagged (calibrated risk {score:.1%}). "
        "No vulnerability review is required unless you have external reason to suspect a bug."
    )


def _mock_window_explanation(function: dict[str, Any], marker: dict[str, Any]) -> str:
    window_index = marker["window_index"]
    window_prob = float(marker.get("window_prob", 0.0))
    if str(marker.get("highlight_kind")) == "safe":
        thresholds = function.get("thresholds") or {}
        window_threshold = float(thresholds.get("window", 0.36))
        return (
            f"Window {window_index} is below the detector threshold "
            f"({window_prob:.1%} vs {window_threshold:.1%}). No review is required for this segment."
        )

    status = str(function["agreement_status"])
    score = float(function.get("function_score_calibrated", 0.0))
    window_index = marker["window_index"]
    if status == "agree_positive":
        return (
            f"Window {window_index} crossed both function-level ({score:.1%}) and window-level thresholds. "
            "Review memory, bounds, and input validation in this segment."
        )
    if status == "review_suggested":
        return (
            f"Window {window_index} exceeded the window detector threshold while the pooled function score "
            f"stayed at {score:.1%}. Localized unsafe logic may be diluted by max-pooling."
        )
    if status == "diffuse_risk":
        return (
            f"Window {window_index} is a max-pool contributor. Function risk is {score:.1%} but no single "
            "window crossed threshold — risk may be distributed across contributors."
        )
    return f"No review required for this window. Calibrated function risk is {score:.1%}."


def _window_record(function: dict[str, Any], window_index: int) -> dict[str, Any]:
    windows = function.get("prompt_windows") or []
    selected = [w for w in windows if int(w["window_index"]) == int(window_index)]
    if not selected:
        for pool in (function.get("flagged_windows") or [], function.get("contributing_windows") or []):
            selected = [w for w in pool if int(w["window_index"]) == int(window_index)]
            if selected:
                break
    record = dict(function)
    record["prompt_windows"] = selected
    record["highlight_window_indices"] = [window_index]
    return record


def _truncate_window_code(window: dict[str, Any], max_chars: int) -> dict[str, Any]:
    trimmed = dict(window)
    code = str(trimmed.get("code", ""))
    if len(code) > max_chars:
        trimmed["code"] = code[:max_chars].rstrip() + "\n// … (truncated for inference)"
    return trimmed


def _llm_scan_settings() -> dict[str, Any]:
    try:
        from services.config_service import load_config

        scan_cfg = load_config().get("scan") or {}
    except Exception:
        scan_cfg = {}
    return {
        "max_new_tokens": int(scan_cfg.get("llm_max_new_tokens", 192)),
        "max_code_chars": int(scan_cfg.get("llm_max_code_chars", 2400)),
        "llm_explanation_enabled": bool(scan_cfg.get("llm_explanation_enabled", True)),
        "llm_chain_of_thought": bool(scan_cfg.get("llm_chain_of_thought", True)),
    }


def _resolve_llm_provider(provider: str) -> tuple[str, Any | None, Any | None]:
    if provider != "huggingface":
        return "mock", None, None

    import sys

    project_root = Path(__file__).resolve().parents[3]
    llm_scripts = project_root / "08_LLM" / "training_scripts"
    xai_scripts = project_root / "07_XAI" / "training_scripts"
    if str(llm_scripts) not in sys.path:
        sys.path.insert(0, str(llm_scripts))
    if str(xai_scripts) not in sys.path:
        sys.path.insert(0, str(xai_scripts))

    try:
        from llm_common import (  # noqa: PLC0415
            generate_explanation,
            generate_verified_explanation,
            load_config as load_llm_config,
            load_model_and_tokenizer,
            model_is_cached,
            model_is_downloaded,
            release_model_cache,
        )
        from grounded_explain import build_grounded_window_prompt  # noqa: PLC0415
        from xai_common import build_single_window_prompt  # noqa: PLC0415

        llm_config = load_llm_config()
        if not model_is_downloaded(llm_config):
            return "mock", None, None
        return (
            "huggingface",
            llm_config,
            (
                generate_explanation,
                generate_verified_explanation,
                build_grounded_window_prompt,
                build_single_window_prompt,
                load_model_and_tokenizer,
                release_model_cache,
            ),
        )
    except Exception:
        return "mock", None, None


def _explain_functions(
    functions: list[dict[str, Any]],
    *,
    provider: str,
    tracker: ScanProgressTracker | None = None,
    explanations_enabled: bool = True,
    chain_of_thought: bool = True,
) -> str:
    scan_settings = _llm_scan_settings()
    if not explanations_enabled:
        if tracker is not None:
            tracker.set_explain_steps(0)
            tracker.complete("LLM explanations disabled")
        return "disabled"

    effective_provider, llm_config, llm_helpers = _resolve_llm_provider(provider)
    if tracker is not None:
        tracker.set_explain_steps(
            _count_explain_steps(
                functions,
                use_llm=effective_provider == "huggingface",
                explanations_enabled=True,
            )
        )

    generate_explanation = None
    generate_verified_explanation = None
    build_grounded_window_prompt = None
    build_single_window_prompt = None
    load_model_and_tokenizer = None
    release_model_cache = None
    if llm_helpers is not None:
        (
            generate_explanation,
            generate_verified_explanation,
            build_grounded_window_prompt,
            build_single_window_prompt,
            load_model_and_tokenizer,
            release_model_cache,
        ) = llm_helpers

    llm_ready = False
    if effective_provider == "huggingface" and load_model_and_tokenizer is not None:
        from llm_common import model_is_cached  # noqa: PLC0415

        release_gpu_for_llm()
        _log_gpu_memory("before Qwen load")
        was_cached = llm_config is not None and model_is_cached(llm_config)
        if was_cached:
            print("LLM: reusing loaded Qwen weights", flush=True)
        else:
            from llm_common import cached_model_source  # noqa: PLC0415

            if cached_model_source():
                print("LLM: switching model — unloading previous weights…", flush=True)
            print("LLM: loading Qwen weights (1–3 min on first load; do not use --reload)…", flush=True)
        try:
            load_model_and_tokenizer(llm_config)
            llm_ready = True
            _log_gpu_memory("after Qwen load")
            if tracker is not None:
                tracker.complete("Local LLM ready (cached)" if was_cached else "Local LLM ready")
        except Exception as exc:
            print(f"LLM load failed ({exc!r}) — using template explanations.", flush=True)
            if release_model_cache is not None:
                release_model_cache()
            effective_provider = "mock"
            llm_ready = False
            if tracker is not None:
                tracker.set_explain_steps(_count_explain_steps(functions, use_llm=False))

    try:
        for index, function in enumerate(functions):
            name = function.get("name", "function")
            markers = function.get("markers") or []
            status = str(function.get("agreement_status", "agree_negative"))

            if effective_provider == "mock" or not llm_ready:
                function["explanation"] = _mock_function_explanation(function)
                for marker in markers:
                    marker["explanation"] = _mock_window_explanation(function, marker)
                if tracker is not None:
                    tracker.complete(f"Explaining {name} (template)")
                continue

            function["explanation"] = _mock_function_explanation(function)

            actionable = [m for m in markers if str(m.get("highlight_kind")) != "safe"]
            for marker in markers:
                if str(marker.get("highlight_kind")) == "safe":
                    marker["explanation"] = _mock_window_explanation(function, marker)

            if status == "agree_negative" or not actionable:
                if tracker is not None:
                    tracker.complete(f"Summarizing {name}")
                continue

            assert generate_explanation is not None and build_single_window_prompt is not None
            grounded_cfg = (llm_config or {}).get("grounded_explanation") or {}
            use_grounded = bool(grounded_cfg.get("enabled", True))
            use_chain_of_thought = bool(chain_of_thought)

            for marker in actionable:
                window_index = int(marker["window_index"])
                window_record = _window_record(function, window_index)
                trimmed = _truncate_window_code(
                    (window_record.get("prompt_windows") or [{}])[0],
                    scan_settings["max_code_chars"],
                )
                window_record["prompt_windows"] = [trimmed]
                grounding_context: dict[str, Any] = {}
                try:
                    if use_grounded and build_grounded_window_prompt is not None:
                        window_prompt, grounding_context = build_grounded_window_prompt(
                            window_record,
                            window_index,
                            chain_of_thought=use_chain_of_thought,
                        )
                    else:
                        window_prompt = build_single_window_prompt(window_record, window_index)

                    if (
                        use_grounded
                        and bool(grounded_cfg.get("verification_pass", True))
                        and generate_verified_explanation is not None
                        and grounding_context.get("window_code")
                    ):
                        verified = generate_verified_explanation(
                            window_prompt,
                            window_code=str(grounding_context["window_code"]),
                            config=llm_config,
                            max_new_tokens=scan_settings["max_new_tokens"],
                            web_scan=True,
                            clear_cache_after=True,
                        )
                        marker["explanation"] = verified["explanation"]
                        marker["explanation_grounding"] = {
                            "detected_cwe": grounding_context.get("detected_cwe"),
                            "pattern_name": grounding_context.get("pattern_name"),
                            "top_tokens": grounding_context.get("top_tokens"),
                            "token_attribution_source": grounding_context.get("token_attribution_source"),
                            "pattern_category": grounding_context.get("pattern_category"),
                            "verified": verified.get("verified"),
                        }
                    else:
                        marker["explanation"] = generate_explanation(
                            window_prompt,
                            llm_config,
                            max_new_tokens=scan_settings["max_new_tokens"],
                            web_scan=True,
                            clear_cache_after=True,
                        )
                        if grounding_context:
                            marker["explanation_grounding"] = {
                                "detected_cwe": grounding_context.get("detected_cwe"),
                                "pattern_name": grounding_context.get("pattern_name"),
                                "top_tokens": grounding_context.get("top_tokens"),
                                "token_attribution_source": grounding_context.get("token_attribution_source"),
                                "pattern_category": grounding_context.get("pattern_category"),
                            }
                except Exception as exc:
                    print(f"LLM explain failed for window {window_index} ({exc!r}) — using template.", flush=True)
                    marker["explanation"] = _mock_window_explanation(function, marker)
                if tracker is not None:
                    tracker.complete(f"Explaining window {window_index} · {name}")
    finally:
        if effective_provider == "huggingface" and release_model_cache is not None:
            web_cfg = (llm_config or {}).get("web_scan") or {}
            if bool(web_cfg.get("unload_model_after_scan", False)):
                release_model_cache()

    return effective_provider if llm_ready else "mock"


def run_scan(
    *,
    source: str,
    filename: str,
    progress: ProgressCallback | None = None,
    llm_provider: str = "mock",
    max_functions: int | None = 50,
) -> ScanResult:
    scan_id = uuid.uuid4().hex[:12]
    uploaded_at = datetime.now(timezone.utc).isoformat()
    phase_log: list[dict[str, Any]] = []
    tracker = ScanProgressTracker(progress)
    scan_options = _llm_scan_settings()
    llm_explanation_enabled = bool(scan_options["llm_explanation_enabled"])
    llm_chain_of_thought = bool(scan_options["llm_chain_of_thought"])

    extracted = extract_functions_from_source(source, filename)
    if max_functions is not None:
        extracted = extracted[: int(max_functions)]
    tracker.set_plan(function_count=max(len(extracted), 1), explain_steps=max(len(extracted), 1))
    tracker.complete(f"Extracted {len(extracted)} function(s)")
    phase_log.append({"phase": "extract", "function_count": len(extracted)})
    if not extracted:
        raise ValueError(
            "No C/C++ function definitions found. "
            "Upload a .c/.cpp file with at least one function body (not just #includes or headers)."
        )

    cleaner, batcher, embedder_mod = get_pipeline_helpers()
    from pipeline.model_runtime import _load_dataset_config

    cfg = _relaxed_clean_cfg(_load_dataset_config())
    batcher_cfg = cfg["9_batcher"]
    safe, high, max_tok = int(batcher_cfg["token_safe"]), int(batcher_cfg["token_high"]), int(batcher_cfg["token_max"])
    tokenizer_name = str(batcher_cfg.get("tokenizer_name", "microsoft/graphcodebert-base"))
    counter = batcher.TokenCounter(tokenizer_name)
    ecfg = cfg["10_embedder"]
    batch_size = int(ecfg.get("batch_size", 16))

    cleaned_functions: list[tuple[ExtractedFunction, list[int]]] = []
    for item in extracted:
        clean_result = cleaner.clean_code_with_tracker(item.code, cfg, base_line=item.start_line)
        if not clean_result.code.strip():
            continue
        cleaned_functions.append(
            (
                ExtractedFunction(
                    id=item.id,
                    name=item.name,
                    code=clean_result.code,
                    start_line=item.start_line,
                    end_line=item.end_line,
                ),
                clean_result.line_map,
            )
        )
    if not cleaned_functions:
        raise ValueError("No functions remained after cleaning. Upload a larger C/C++ function body.")
    tracker.set_plan(function_count=len(cleaned_functions), explain_steps=len(cleaned_functions))
    tracker.complete(f"Cleaned {len(cleaned_functions)} function(s)")
    phase_log.append({"phase": "clean", "function_count": len(cleaned_functions)})

    windowed_rows: list[dict[str, Any]] = []
    for item, line_map in cleaned_functions:
        row = {
            "id": item.id,
            "code": item.code,
            "label": 0,
            "func_name": item.name,
            "file_path": filename,
        }
        windowed_rows.extend(batcher.expand_row(row, counter, safe, high, max_tok))
    tracker.complete(f"Split into {len(windowed_rows)} window(s)")
    phase_log.append({"phase": "window", "window_count": len(windowed_rows)})
    _release_token_counter(counter)
    del counter

    codes = [str(row["code"]) for row in windowed_rows]
    embeddings, parse_coverages, token_counts = encode_scan_windows(codes, batch_size=batch_size)
    tracker.complete("GraphCodeBERT encoding complete")
    del codes
    phase_log.append({"phase": "embed", "embedding_shape": list(embeddings.shape)})

    bundle = get_model_bundle()

    by_function: dict[str, list[int]] = {}
    for index, row in enumerate(windowed_rows):
        group_id = str(row["function_group_id"])
        by_function.setdefault(group_id, []).append(index)

    results: list[dict[str, Any]] = []
    for item, line_map in cleaned_functions:
        indices = by_function.get(item.id, [])
        if not indices:
            continue
        rows = [windowed_rows[i] for i in indices]
        window_embeddings = embeddings[indices]

        scored = score_function_windows(
            bundle,
            function_group_id=item.id,
            window_rows=rows,
            window_embeddings=window_embeddings,
            on_step_complete=tracker.complete,
            function_name=item.name,
        )
        scored["name"] = item.name
        scored["full_code"] = item.code
        scored["original_code"] = "\n".join(source.splitlines()[item.start_line - 1 : item.end_line])
        scored["file_start_line"] = item.start_line
        scored["file_end_line"] = item.end_line
        scored["line_map"] = line_map
        scored["code_start_line"] = line_map[0] if line_map else item.start_line
        scored["filename"] = filename
        scored = attach_signature_attribution(
            scored,
            function_threshold=float(bundle.function_threshold),
        )
        scored["markers"] = _build_markers(
            scored,
            item.code,
            line_map=line_map,
            file_end_line=item.end_line,
        )
        results.append(scored)

    shap_windows = 0
    if llm_explanation_enabled:
        try:
            xai_scripts = Path(__file__).resolve().parents[3] / "07_XAI" / "training_scripts"
            if str(xai_scripts) not in sys.path:
                sys.path.insert(0, str(xai_scripts))
            from shap_token_attribution import attach_shap_tokens_to_functions  # noqa: PLC0415

            llm_config_for_shap: dict[str, Any] | None = None
            llm_config_path = Path(__file__).resolve().parents[3] / "08_LLM" / "llm_config.yaml"
            if llm_config_path.exists():
                import yaml

                llm_config_for_shap = yaml.safe_load(llm_config_path.read_text(encoding="utf-8"))

            encoder = get_encoder()
            shap_windows = attach_shap_tokens_to_functions(
                results,
                encoder=encoder,
                bundle=bundle,
                settings=llm_config_for_shap,
                on_step_complete=tracker.complete,
            )
            phase_log.append({"phase": "shap_tokens", "window_count": shap_windows})
            tracker.complete(f"SHAP token attribution · {shap_windows} window(s)")
        except Exception as exc:
            print(f"SHAP attribution skipped ({exc!r})", flush=True)
            phase_log.append({"phase": "shap_tokens", "error": repr(exc)})

    release_encoder()

    scoring_mode = bundle.scoring_mode
    spread_weight = float(bundle.spread_weight)
    thresholds = {
        "function": float(bundle.function_threshold),
        "function_review": float(bundle.function_review_threshold),
        "window": float(bundle.window_threshold),
        "scoring_mode": scoring_mode,
    }
    release_model_bundle(bundle)
    del bundle
    del embeddings
    del windowed_rows
    tracker.complete("Signature pattern scan complete")
    phase_log.append({"phase": "signature_match", "function_count": len(results)})

    explain_provider = _explain_functions(
        results,
        provider=llm_provider,
        tracker=tracker,
        explanations_enabled=llm_explanation_enabled,
        chain_of_thought=llm_chain_of_thought,
    )
    tracker.complete("Finalizing results")
    file_markers = merge_file_markers(results)
    file_score = build_file_score(
        results,
        threshold=thresholds["function"],
        weight=spread_weight,
    )
    phase_log.append({
        "phase": "explain",
        "provider": explain_provider,
        "requested_provider": llm_provider,
        "llm_explanation_enabled": llm_explanation_enabled,
        "llm_chain_of_thought": llm_chain_of_thought,
    })

    return ScanResult(
        scan_id=scan_id,
        filename=filename,
        uploaded_at=uploaded_at,
        source_code=source,
        functions=results,
        file_markers=file_markers,
        file_score=file_score,
        phase_log=phase_log,
        thresholds=thresholds,
        llm_explanation_enabled=llm_explanation_enabled,
        llm_chain_of_thought=llm_chain_of_thought,
    )
