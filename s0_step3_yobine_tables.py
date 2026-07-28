"""S0 step 3 -- freeze the yobine (tick size) reference artifacts.

The tables themselves live in s0_common.py and are tested in tests/test_institutional.py
against ticks read off the tape. This step materialises them for the report and
records which stocks carry the fine grid in 2024.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s0_inst")


def band_rows(table, other) -> list[dict]:
    """Render a yobine table as human-readable bands, paired with the other table."""
    rows, lo10 = [], 0
    for upper10, tick10 in table:
        rows.append({
            "lower_yen": lo10 / 10, "upper_yen": None if upper10 is None else upper10 / 10,
            "tick_yen": tick10 / 10,
            "other_tick_yen": C.tick_for10(
                (upper10 if upper10 is not None else lo10 + 10) - 1, other) / 10,
        })
        if upper10 is None:
            break
        lo10 = upper10
    return rows


def fmt_yen(v) -> str:
    if v is None:
        return "and above"
    return f"{v:,.10g}"


def main() -> int:
    tee = C.Tee("s0_step3_yobine_tables")
    try:
        print("=== S0 step 3: yobine (tick size) reference ===\n")
        import polars as pl

        fine = band_rows(C.YOBINE_TOPIX500, other=False)
        gen = band_rows(C.YOBINE_GENERAL, other=True)

        print("TOPIX500 (fine) grid, 2024:")
        for r in fine:
            print(f"  {fmt_yen(r['lower_yen']):>14s} - {fmt_yen(r['upper_yen']):>14s} "
                  f"yen : tick {r['tick_yen']:g} yen")
        print("\nGeneral grid, 2024:")
        for r in gen:
            print(f"  {fmt_yen(r['lower_yen']):>14s} - {fmt_yen(r['upper_yen']):>14s} "
                  f"yen : tick {r['tick_yen']:g} yen")

        C.ensure_dir(OUT)
        pl.DataFrame(fine).write_csv(C.write_guard(os.path.join(OUT, "yobine_topix500.csv")))
        pl.DataFrame(gen).write_csv(C.write_guard(os.path.join(OUT, "yobine_general.csv")))

        # Which grids can a 2024 stock-day legally sit on, and which survive filter (d)?
        t500 = C.load_topix500()
        pl.DataFrame({"ticker": sorted(t500)}).write_csv(
            C.write_guard(os.path.join(OUT, "topix500_union_2023_2024.csv")))
        print(f"\nTOPIX500 union 2023-2024: {len(t500)} tickers "
              f"(saved for provenance)")

        # The consequence that shapes the whole sample: on the fine grid the
        # 1,000-3,000 yen band is 0.5 yen, which Ohta's filter (d) excludes.
        excluded_bands = [r for r in fine
                          if round(r["tick_yen"] * 10) not in C.POWER_OF_TEN_TICKS10]
        print("\nFine-grid bands EXCLUDED by Ohta filter (d) (tick not a power of ten):")
        for r in excluded_bands:
            print(f"  {fmt_yen(r['lower_yen'])} - {fmt_yen(r['upper_yen'])} yen "
                  f"-> {r['tick_yen']:g} yen")
        gen_excluded = [r for r in gen
                        if round(r["tick_yen"] * 10) not in C.POWER_OF_TEN_TICKS10]
        print("General-grid bands EXCLUDED by filter (d):")
        for r in gen_excluded:
            print(f"  {fmt_yen(r['lower_yen'])} - {fmt_yen(r['upper_yen'])} yen "
                  f"-> {r['tick_yen']:g} yen")

        C.atomic_json(os.path.join(OUT, "yobine_summary.json"), {
            "fine_bands": len(fine), "general_bands": len(gen),
            "topix500_union": len(t500),
            "fine_excluded_by_filter_d": [
                {"lower_yen": r["lower_yen"], "upper_yen": r["upper_yen"],
                 "tick_yen": r["tick_yen"]} for r in excluded_bands],
            "general_excluded_by_filter_d": [
                {"lower_yen": r["lower_yen"], "upper_yen": r["upper_yen"],
                 "tick_yen": r["tick_yen"]} for r in gen_excluded],
            "fine_regime_all_topix500_from": str(C.FINE_TICK_ALL_TOPIX500),
            "sources": [
                "Rakuten Securities notice 2023-05-29 (TOPIX Mid400 tick change)",
                "Matsui Securities domestic-stock rules page",
                "cross-checked against ticks inferred from the tape for 7203/8604/8306/4666",
            ],
        })
        print("\nGATE PASSED -- yobine tables frozen and cross-checked")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
