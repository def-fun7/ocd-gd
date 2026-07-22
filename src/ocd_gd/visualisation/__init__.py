"""
Visualization subpackage for galactic orbit chaos detection.
"""

from .matplotlib_backend import (
    plot_sali_mpl,
    plot_gali_mpl,
    plot_trajectory_2d_mpl,
    plot_trajectory_3d_mpl,
    plot_phase_space_mpl,
    plot_energy_drift_mpl,
    plot_colored_trajectory_2d_mpl,
)

from .plotly_backend import (
    plot_sali_plotly,
    plot_gali_plotly,
    plot_trajectory_2d_plotly,
    plot_trajectory_3d_plotly,
    plot_phase_space_plotly,
    plot_energy_drift_plotly,
    plot_colored_trajectory_2d_plotly,
)

from .dashboard import (
    plot_dashboard_mpl,
    plot_dashboard_plotly,
)

from .batch import (
    plot_sali_batch_mpl,
    plot_gali_batch_mpl,
    plot_sali_gali_dual_batch_mpl,
)
from .utils import set_publication_style, set_output_dir

__all__ = [
    # Matplotlib
    "plot_sali_mpl",
    "plot_gali_mpl",
    "plot_trajectory_2d_mpl",
    "plot_trajectory_3d_mpl",
    "plot_phase_space_mpl",
    "plot_energy_drift_mpl",
    "plot_colored_trajectory_2d_mpl",
    # Plotly
    "plot_sali_plotly",
    "plot_gali_plotly",
    "plot_trajectory_2d_plotly",
    "plot_trajectory_3d_plotly",
    "plot_phase_space_plotly",
    "plot_energy_drift_plotly",
    "plot_colored_trajectory_2d_plotly",
    # Dashboards & Utils
    "plot_dashboard_mpl",
    "plot_dashboard_plotly",
    "set_publication_style",
    "set_output_dir",
    # batch
    "plot_sali_batch_mpl",
    "plot_gali_batch_mpl",
    "plot_sali_gali_dual_batch_mpl",
]
