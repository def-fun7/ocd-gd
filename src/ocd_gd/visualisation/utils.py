"""
Styling defaults and utility functions for visualization.
"""

__all__ = ["set_output_dir", "set_publication_style"]

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Global tracking for export directories
_OUTPUT_DIRS = {
    "root": Path("./plots"),
    "matplotlib": Path("./plots/matplotlib"),
    "plotly": Path("./plots/plotly"),
}


def set_output_dir(path: str = "./plots") -> tuple[Path, Path]:
    """Set default base directory for plot exports and create backend subfolders.

    Returns
    -------
    Tuple[Path, Path]
        Paths to (matplotlib_dir, plotly_dir)
    """
    root = Path(path)
    mpl_dir = root / "matplotlib"
    plotly_dir = root / "plotly"

    mpl_dir.mkdir(parents=True, exist_ok=True)
    plotly_dir.mkdir(parents=True, exist_ok=True)

    _OUTPUT_DIRS["root"] = root
    _OUTPUT_DIRS["matplotlib"] = mpl_dir
    _OUTPUT_DIRS["plotly"] = plotly_dir

    return mpl_dir, plotly_dir


def resolve_save_path(save_path: str | None, backend: str) -> str | None:
    """Helper to route relative save filenames into designated backend folders."""
    if save_path is None:
        return None

    path_obj = Path(save_path)
    # If absolute path or user explicitly specified a folder, don't overwrite it
    if path_obj.is_absolute() or len(path_obj.parts) > 1:
        return save_path

    # Make sure output directories exist even if user didn't call set_output_dir()
    set_output_dir(_OUTPUT_DIRS["root"])

    return str(_OUTPUT_DIRS[backend] / save_path)


# Default style parameters across Matplotlib plots
MPL_STYLE_DEFAULTS: dict[str, Any] = {
    # Figure properties
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "figure.figsize": (8, 6),
    "savefig.dpi": 600,
    # Axes & spine styling
    "axes.facecolor": "white",
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.grid": True,
    # Fonts
    "font.family": "sans-serif",
    "font.size": 10,
    # Ticks
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    # Grid
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    # Legend
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    # Lines, markers, and errorbars
    "lines.linewidth": 1.0,
    "lines.markersize": 6,
    "errorbar.capsize": 5,
}


def set_publication_style():
    """Apply default scientific plotting style to Matplotlib globally."""
    plt.rcParams.update(MPL_STYLE_DEFAULTS)


# Common color palettes
PALETTES = {
    "sali": "crimson",
    "gali": "plasma",
    "trajectory": "navy",
    "3d_trajectory": "teal",
    "phase_space": "purple",
    "energy": "darkgreen",
}
