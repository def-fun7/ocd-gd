"""
Styling defaults and utility functions for visualization.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt

import os
from pathlib import Path
from typing import Optional, Tuple

# Global tracking for export directories
_OUTPUT_DIRS = {
    "root": Path("./plots"),
    "matplotlib": Path("./plots/matplotlib"),
    "plotly": Path("./plots/plotly"),
}


def set_output_dir(path: str = "./plots") -> Tuple[Path, Path]:
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


def resolve_save_path(save_path: Optional[str], backend: str) -> Optional[str]:
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
MPL_STYLE_DEFAULTS: Dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
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
