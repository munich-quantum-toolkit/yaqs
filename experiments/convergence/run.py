# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Orchestrate TFIM TDVP substep convergence audit."""

from __future__ import annotations

import argparse

from config import OUTPUT_DIR, apply_thread_limits, production_config, config_hash
from generate_data import finalize, generate
from plot import main as plot_main
from store import save_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TFIM TDVP substep convergence audit.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Finalize tables/figure from DB/CSVs.")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)

    apply_thread_limits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "config.json", {**production_config(), "config_hash": config_hash()})

    if args.plot_only or args.finalize_only:
        finalize()
        return plot_main([])

    print("=== TDVP substep convergence generation ===", flush=True)
    summary = generate(resume=not args.no_resume)
    print(summary["verdict"], flush=True)
    return plot_main([])


if __name__ == "__main__":
    raise SystemExit(main())
