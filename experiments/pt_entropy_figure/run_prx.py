"""CLI entry point for the PRX Quantum single-column figure variant."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_DATA_CSV, DEFAULT_OUTPUT_DIR, SPECTRUM_JS
from .data import build_panel_b_curves, load_entropy_table
from .layouts import PRX_SINGLE_COLUMN_LAYOUT
from .plot import _panel_a_ylim, build_figure, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the PRX Quantum single-column variant of the S_V versus S_MPO figure "
            "(layout inspired by cut_vs_j.py)."
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
    parser.add_argument("--png-dpi", type=int, default=PRX_SINGLE_COLUMN_LAYOUT.png_dpi)
    args = parser.parse_args()

    layout = PRX_SINGLE_COLUMN_LAYOUT
    points = load_entropy_table(args.data_csv.resolve())
    curves = build_panel_b_curves()
    print(
        f"PRX single-column layout: {layout.figsize[0]:.3f} x {layout.figsize[1]:.3f} in",
        flush=True,
    )
    print(
        f"Panel (b): {len(curves)} spectra "
        f"({len(SPECTRUM_JS)} couplings x 2 quantities at c=2)",
        flush=True,
    )
    fig = build_figure(points, curves, layout=layout)
    y_lo, y_hi = _panel_a_ylim(points)
    print(f"Panel (a) y-limits: [{y_lo:.3e}, {y_hi:.3e}] nats", flush=True)
    pdf_path, png_path = save_figure(
        fig,
        args.output_dir.resolve(),
        stem=layout.output_stem,
        dpi=int(args.png_dpi),
        pad_inches=layout.savefig_pad_inches,
    )
    import matplotlib.pyplot as plt
    import shutil

    plt.close(fig)
    root_dir = Path(__file__).resolve().parents[1]
    root_pdf = root_dir / pdf_path.name
    root_png = root_dir / png_path.name
    shutil.copy2(pdf_path, root_pdf)
    shutil.copy2(png_path, root_png)
    print(f"Wrote {pdf_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    print(f"Copied {root_pdf}", flush=True)
    print(f"Copied {root_png}", flush=True)


if __name__ == "__main__":
    main()
