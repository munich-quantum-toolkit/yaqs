"""CLI entry point for the S_V versus S_MPO comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_DATA_CSV, DEFAULT_OUTPUT_DIR, SPECTRUM_JS
from .data import build_panel_b_curves, load_entropy_table
from .plot import _panel_a_ylim, build_figure, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a two-panel figure comparing response entropy S_V with "
            "causal-block MPO bond entropy S_MPO."
        )
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DEFAULT_DATA_CSV,
        help="Bundled entropy table for panel (a).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PDF/PNG exports.",
    )
    parser.add_argument("--png-dpi", type=int, default=400)
    args = parser.parse_args()

    points = load_entropy_table(args.data_csv.resolve())
    curves = build_panel_b_curves()
    print(f"Panel (b): {len(curves)} spectra "
          f"({len(SPECTRUM_JS)} couplings x 2 quantities at c=2)")
    fig = build_figure(points, curves)
    y_lo, y_hi = _panel_a_ylim(points)
    print(f"Panel (a) y-limits: [{y_lo:.3e}, {y_hi:.3e}] nats")
    pdf_path, png_path = save_figure(fig, args.output_dir.resolve(), dpi=int(args.png_dpi))
    import matplotlib.pyplot as plt
    import shutil

    plt.close(fig)
    # Convenience copy at experiments/ root for quick browsing.
    root_dir = Path(__file__).resolve().parents[1]
    root_pdf = root_dir / pdf_path.name
    root_png = root_dir / png_path.name
    shutil.copy2(pdf_path, root_pdf)
    shutil.copy2(png_path, root_png)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Copied {root_pdf}")
    print(f"Copied {root_png}")


if __name__ == "__main__":
    main()
