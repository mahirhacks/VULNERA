"""Extract C/C++ functions from an uploaded source file."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ExtractedFunction:
    id: str
    name: str
    code: str
    start_line: int
    end_line: int


_FUNC_NAME = re.compile(
    r"(?:[\w:\*&<>\s~]+?)"
    r"[\*\&]*\s*"
    r"(\w+)\s*\([^;]*\)\s*(?:const\s*)?"
    r"(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?\{?",
)
_PREPROC_LINE = re.compile(r"^\s*#\s*(?:include|import|define|undef|pragma|line|error|warning|if|ifdef|ifndef|elif|else|endif)\b")
_SKIP_LINE = re.compile(r"^\s*(?://|/\*|\*)")
_CONTROL_FLOW_LEAD = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "catch",
        "try",
        "return",
        "goto",
        "break",
        "continue",
        "sizeof",
        "namespace",
        "class",
        "struct",
        "union",
        "enum",
        "typedef",
        "template",
        "using",
    }
)


def _function_name_from_header(before_brace: str) -> str | None:
    """Extract the defined function's name from text before the opening brace."""
    open_paren = before_brace.find("(")
    if open_paren < 0:
        return None
    pre = before_brace[:open_paren].strip()
    if not pre:
        return None
    pre = re.sub(r"[\*\&\s]+$", "", pre).strip()
    tokens = re.findall(r"[A-Za-z_]\w*", pre)
    return tokens[-1] if tokens else None


def _guess_name(header_line: str, fallback: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", header_line, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE).strip()
    brace = text.find("{")
    before = text[:brace].strip() if brace >= 0 else text
    return _function_name_from_header(before) or fallback


def _is_skippable_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _PREPROC_LINE.match(line) or stripped.startswith("#"):
        return True
    if _SKIP_LINE.match(line):
        return True
    return False


def _strip_preprocessor_lines(code: str) -> str:
    """Remove preprocessor / include directives from an extracted body."""
    kept: list[str] = []
    for line in code.splitlines():
        if _PREPROC_LINE.match(line) or line.strip().startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _leading_identifier(header_before_paren: str) -> str | None:
    text = re.sub(r"/\*.*?\*/", "", header_before_paren, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE).strip()
    if not text:
        return None
    tokens = re.findall(r"[A-Za-z_]\w*", text)
    if not tokens:
        return None
    return tokens[-1]


def _is_function_definition_header(header: str) -> bool:
    """
  Return True when the header looks like a function definition, not a struct,
  namespace, control-flow block, or initializer.
  """
    text = " ".join(line.strip() for line in header.splitlines() if line.strip())
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE).strip()
    if not text:
        return False

    brace = text.find("{")
    if brace < 0:
        return False

    before_brace = text[:brace].strip()
    if not before_brace:
        return False

    open_paren = before_brace.rfind("(")
    close_paren = before_brace.rfind(")")
    if open_paren < 0 or close_paren < open_paren:
        return False

    pre_param = before_brace[:open_paren].strip()
    if not pre_param:
        return False

    lowered = pre_param.lower()
    if lowered.startswith("extern ") and '"' in pre_param and "(" not in pre_param.replace('"', ""):
        return False

    lead = _leading_identifier(pre_param)
    if lead and lead.lower() in _CONTROL_FLOW_LEAD:
        return False

    # typedef struct / bare aggregate definitions
    preview = lowered
    if preview.startswith("typedef struct") or preview.startswith("typedef union") or preview.startswith("typedef enum"):
        return False
    if preview.startswith("struct ") and "(" not in preview:
        return False
    if preview.startswith("class ") and "(" not in preview:
        return False
    if preview.startswith("namespace ") and "(" not in preview:
        return False

    name = _function_name_from_header(before_brace)
    if not name:
        return False
    if name.lower() in _CONTROL_FLOW_LEAD:
        return False

    return bool(_FUNC_NAME.search(before_brace + "{")) or name.isidentifier()


def extract_functions_from_source(source: str, filename: str) -> list[ExtractedFunction]:
    """Brace-matching extractor that returns function bodies only (no includes/macros)."""
    lines = source.splitlines()
    if not lines:
        return []

    functions: list[ExtractedFunction] = []
    index = 0
    func_idx = 0
    stem = Path(filename).stem

    while index < len(lines):
        if _is_skippable_line(lines[index]):
            index += 1
            continue

        header_start = index
        cursor = index
        brace_found = False

        while cursor < len(lines):
            if _is_skippable_line(lines[cursor]):
                cursor += 1
                continue
            if "{" in lines[cursor]:
                brace_found = True
                break
            if ";" in lines[cursor] and "{" not in lines[cursor]:
                break
            cursor += 1

        if not brace_found:
            index += 1
            continue

        header_text = "\n".join(
            lines[i]
            for i in range(header_start, cursor + 1)
            if not _is_skippable_line(lines[i])
        )
        if not _is_function_definition_header(header_text):
            index = cursor + 1
            continue

        brace_count = 0
        end_cursor = cursor
        while end_cursor < len(lines):
            brace_count += lines[end_cursor].count("{") - lines[end_cursor].count("}")
            if brace_count == 0 and end_cursor >= cursor:
                break
            end_cursor += 1

        if brace_count != 0:
            index += 1
            continue

        func_idx += 1
        raw_code = "\n".join(lines[header_start : end_cursor + 1])
        code = _strip_preprocessor_lines(raw_code)
        if not code.strip():
            index = end_cursor + 1
            continue

        header_line = next(
            (lines[i] for i in range(header_start, cursor + 1) if not _is_skippable_line(lines[i])),
            lines[header_start],
        )
        functions.append(
            ExtractedFunction(
                id=f"{stem}_fn{func_idx}",
                name=_guess_name(header_line, f"function_{func_idx}"),
                code=code,
                start_line=header_start + 1,
                end_line=end_cursor + 1,
            )
        )
        index = end_cursor + 1

    return functions
