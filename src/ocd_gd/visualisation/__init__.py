"""
Visualization subpackage for galactic orbit chaos detection.
"""

__all__ = [

    "plot_chaos_maps_mpl",
    "plot_chaos_maps_plotly",
    "plot_colored_trajectory_2d_mpl",
    "plot_colored_trajectory_2d_plotly",
    "plot_composite_chaos_map_mpl",
    "plot_composite_chaos_map_plotly",
    "plot_consensus_chaos_map_mpl",
    "plot_consensus_chaos_map_plotly",

    "plot_dashboard_mpl",
    "plot_dashboard_plotly",
    "plot_energy_drift_mpl",
    "plot_energy_drift_plotly",
    "plot_gali_batch_mpl",
    "plot_gali_mpl",
    "plot_gali_plotly",
    "plot_phase_space_mpl",
    "plot_phase_space_plotly",

    "plot_sali_batch_mpl",
    "plot_sali_gali_dual_batch_mpl",

    "plot_sali_mpl",

    "plot_sali_plotly",
    "plot_trajectory_2d_mpl",
    "plot_trajectory_2d_plotly",
    "plot_trajectory_3d_mpl",
    "plot_trajectory_3d_plotly",
    "set_output_dir",
    "set_publication_style",
]


from .batch import (
    plot_gali_batch_mpl,
    plot_sali_batch_mpl,
    plot_sali_gali_dual_batch_mpl,
)
from .dashboard import (
    plot_dashboard_mpl,
    plot_dashboard_plotly,
)
from .grid_mpl_backend import (
    plot_chaos_maps_mpl,
    plot_composite_chaos_map_mpl,
    plot_consensus_chaos_map_mpl,
)
from .grid_plotly_backend import (
    plot_chaos_maps_plotly,
    plot_composite_chaos_map_plotly,
    plot_consensus_chaos_map_plotly,
)
from .mpl_backend import (
    plot_colored_trajectory_2d_mpl,
    plot_energy_drift_mpl,
    plot_gali_mpl,
    plot_phase_space_mpl,
    plot_sali_mpl,
    plot_trajectory_2d_mpl,
    plot_trajectory_3d_mpl,
)
from .plotly_backend import (
    plot_colored_trajectory_2d_plotly,
    plot_energy_drift_plotly,
    plot_gali_plotly,
    plot_phase_space_plotly,
    plot_sali_plotly,
    plot_trajectory_2d_plotly,
    plot_trajectory_3d_plotly,
)
from .utils import set_output_dir, set_publication_style
