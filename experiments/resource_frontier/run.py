# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Orchestrate the 4×4 TFIM resource-frontier experiment."""

from __future__ import annotations

import argparse

from build_frontier import build_all
from config import OUTPUT_DIR, apply_thread_limits, production_config
from generate_data import generate_main_runs, run_timing_repeats, select_timing_candidates
from plot import main as plot_main
from store import FrontierStore, save_json
from validate_report import write_validation
from working_memory import run_validation as run_working_memory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run resource-frontier TFIM pipeline.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-working-memory", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)

    apply_thread_limits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "config.json", production_config())

    if args.plot_only:
        build_all()
        write_validation()
        return plot_main([])

    print("=== Main trajectory generation ===", flush=True)
    summary = generate_main_runs(resume=not args.no_resume)
    print(summary, flush=True)

    store = FrontierStore(OUTPUT_DIR / "raw_runs.sqlite")
    raw = store.fetch_steps(tag="main")
    store.close()

    if not args.skip_timing:
        print("=== Controlled timing repeats ===", flush=True)
        cands = select_timing_candidates(raw)
        print(f"Candidates: {cands}", flush=True)
        run_timing_repeats(cands)

    print("=== Build frontiers ===", flush=True)
    build_all()

    if not args.skip_working_memory:
        print("=== Working-memory validation ===", flush=True)
        run_working_memory()

    print("=== Validation report + figure ===", flush=True)
    write_validation()
    return plot_main([])


if __name__ == "__main__":
    raise SystemExit(main())
