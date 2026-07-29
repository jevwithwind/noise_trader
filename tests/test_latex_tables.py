"""LaTeX table writing must never emit a ragged table.

A tabular whose rows carry a different number of ampersands than its column
specification is a hard LaTeX error, and it surfaces only at build time --- after
the analysis has run. These tests pin the shape.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s4_common as S4


def render(header, rows, notes=""):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.tex")
        # write_guard only permits the project tree, so call the writer's body by
        # pointing it at a file inside the project's own results area instead.
        import s0_common as C
        p = os.path.join(C.RESULTS, "_test_table.tex")
        S4.latex_table(p, "Caption", "tab:test", header, rows, notes=notes)
        out = open(p, encoding="utf-8").read()
        os.remove(p)
        return out


def body_rows(tex: str) -> list[str]:
    inside = tex.split("\\midrule")[1].split("\\bottomrule")[0]
    return [l for l in inside.strip().splitlines() if l.strip()]


def test_every_row_has_the_column_count_the_header_declares():
    header = ["A", "B", "C"]
    rows = [["1", "2", "3"], ["4", "5", "6"]]
    tex = render(header, rows)
    assert "\\begin{tabular}{@{}lrr@{}}" in tex
    for line in body_rows(tex):
        assert line.count("&") == len(header) - 1, line


def test_none_becomes_an_empty_cell_not_the_word_none():
    tex = render(["A", "B"], [["x", None]])
    assert "None" not in tex
    assert body_rows(tex)[0].strip().startswith("x &")


def test_notes_produce_a_threeparttable():
    tex = render(["A"], [["1"]], notes="A note.")
    assert "\\begin{threeparttable}" in tex and "\\end{threeparttable}" in tex
    assert "A note." in tex


def test_no_notes_means_no_threeparttable():
    tex = render(["A"], [["1"]])
    assert "threeparttable" not in tex


def test_caption_precedes_the_tabular():
    tex = render(["A"], [["1"]])
    assert tex.index("\\caption") < tex.index("\\begin{tabular}")


def test_ragged_rows_are_the_caller_s_job_but_are_visible():
    """A short row produces fewer ampersands -- documents why callers pad."""
    tex = render(["A", "B", "C"], [["1", "2", "3"], ["4", "5"]])
    counts = {l.count("&") for l in body_rows(tex)}
    assert counts == {2, 1}, "a short row really does under-fill the tabular"
