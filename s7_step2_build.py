"""S7 step 2 -- build the report and check the log for anything unresolved."""
from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

REPORT = C.REPORT


def main() -> int:
    tee = C.Tee("s7_step2_build")
    try:
        print("=== S7 step 2: build the report ===\n")

        # Every \input and \includegraphics target must exist, or LaTeX quietly
        # produces a PDF with a hole in it.
        missing = []
        for root, _, files in os.walk(REPORT):
            for f in files:
                if not f.endswith(".tex"):
                    continue
                p = os.path.join(root, f)
                src = open(p, encoding="utf-8").read()
                for m in re.finditer(r"\\input\{([^}]+)\}", src):
                    t = m.group(1)
                    cand = [os.path.join(REPORT, t), os.path.join(REPORT, t + ".tex"),
                            os.path.join(root, t), os.path.join(root, t + ".tex")]
                    if not any(os.path.exists(c) for c in cand):
                        missing.append(f"{f}: \\input{{{t}}}")
                for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
                    t = m.group(1)
                    cand = [os.path.join(REPORT, "figures", t),
                            os.path.join(REPORT, t)]
                    if not any(os.path.exists(c) for c in cand):
                        missing.append(f"{f}: \\includegraphics{{{t}}}")
        if missing:
            print(f"{len(missing)} missing inputs:")
            for m in sorted(set(missing)):
                print("  -", m)
            print()

        # latexmk remembers a failed run and will refuse to retry, so a build that
        # follows a fixed error has to be forced.
        for ext in ("aux", "fdb_latexmk", "fls", "bcf", "run.xml", "toc", "out",
                    "bbl", "blg"):
            p = os.path.join(REPORT, f"main.{ext}")
            if os.path.exists(p):
                os.remove(p)
        r = subprocess.run(["latexmk", "-pdf", "-g", "-interaction=nonstopmode",
                            "main.tex"],
                           cwd=REPORT, capture_output=True, text=True)
        log_path = os.path.join(REPORT, "main.log")
        log = open(log_path, encoding="utf-8", errors="ignore").read() \
            if os.path.exists(log_path) else ""

        pdf = os.path.join(REPORT, "main.pdf")
        ok = os.path.exists(pdf) and r.returncode == 0
        print(f"latexmk exit {r.returncode}   pdf {'present' if os.path.exists(pdf) else 'MISSING'}")
        if not ok:
            tail = [l for l in r.stdout.splitlines() if l.strip()][-25:]
            print("\n".join(tail))
            errs = re.findall(r"^! .*$", log, re.M)
            for e in errs[:10]:
                print("  ", e)

        # Unresolved cross-references print as ?? in the PDF.
        undef_ref = len(re.findall(r"Reference `[^']+' on page \d+ undefined", log))
        undef_cit = len(re.findall(r"Citation `[^']+' on page \d+ undefined", log))
        over = len(re.findall(r"Overfull \\hbox", log))
        print(f"undefined references: {undef_ref}   undefined citations: {undef_cit}"
              f"   overfull boxes: {over}")

        if os.path.exists(pdf):
            size = os.path.getsize(pdf) / 1024
            # Modern pdfTeX compresses the page tree into object streams, so the
            # page count is read from the log rather than scraped from the file.
            m = re.findall(r"Output written on main\.pdf \((\d+) pages?", log)
            pages = m[-1] if m else "?"
            print(f"pdf: {pages} pages, {size:.0f} KB")

        if undef_ref or undef_cit:
            print("\nGATE FAILED: the PDF contains unresolved references")
            return 1
        if not ok:
            print("\nGATE FAILED: build did not complete")
            return 1
        print("\nGATE PASSED -- report built cleanly")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
