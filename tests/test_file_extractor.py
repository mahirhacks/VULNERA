"""Tests for web upload C/C++ function extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACK_END = Path(__file__).resolve().parents[1] / "09_WEB" / "back_end"
if str(BACK_END) not in sys.path:
    sys.path.insert(0, str(BACK_END))

from pipeline.file_extractor import extract_functions_from_source  # noqa: E402

TEST_DIR = Path(__file__).resolve().parents[1] / "10_TEST"


def test_extracts_only_function_body_from_test_corpus() -> None:
    source = (TEST_DIR / "test_1.c").read_text(encoding="utf-8")
    functions = extract_functions_from_source(source, "test_1.c")
    assert len(functions) == 1
    fn = functions[0]
    assert fn.name == "tls_process_heartbeat"
    assert "#include" not in fn.code
    assert "typedef struct" not in fn.code
    assert "memcpy(out, hb->buf" in fn.code


def test_pointer_return_type_functions() -> None:
    source = (TEST_DIR / "test_39.c").read_text(encoding="utf-8")
    functions = extract_functions_from_source(source, "test_39.c")
    assert len(functions) == 1
    assert functions[0].name == "alloc_pixel_buffer"
    assert "#include" not in functions[0].code


def test_safe_pointer_return_type() -> None:
    source = (TEST_DIR / "test_40.c").read_text(encoding="utf-8")
    functions = extract_functions_from_source(source, "test_40.c")
    assert len(functions) == 1
    assert functions[0].name == "alloc_pixel_buffer"


def test_ignores_include_only_file() -> None:
    source = "#include <stdio.h>\n#include <string.h>\n"
    assert extract_functions_from_source(source, "headers.c") == []


def test_ignores_extern_c_wrapper_block() -> None:
    source = """
#include <string.h>
extern "C" {
void real_fn(int x) {
    return;
}
}
"""
    functions = extract_functions_from_source(source, "wrap.c")
    assert len(functions) == 1
    assert functions[0].name == "real_fn"
    assert "#include" not in functions[0].code


def test_ignores_struct_and_keeps_function() -> None:
    source = """
#include <stdlib.h>
typedef struct {
    int x;
} item_t;

static int add(item_t *a, item_t *b) {
    return a->x + b->x;
}
"""
    functions = extract_functions_from_source(source, "mix.c")
    assert len(functions) == 1
    assert functions[0].name == "add"
    assert "#include" not in functions[0].code
    assert "typedef" not in functions[0].code


@pytest.mark.parametrize(
    "snippet",
    [
        "if (x) { do_something(); }",
        "for (int i = 0; i < n; i++) { sum += i; }",
        "while (p) { p = p->next; }",
    ],
)
def test_does_not_extract_control_flow_blocks(snippet: str) -> None:
    assert extract_functions_from_source(snippet, "ctrl.c") == []
