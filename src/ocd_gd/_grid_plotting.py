"""
Plotting mixin for GridChaosDetector.

Isolated from grid_detector.py for the same reason `_plotting.py` is isolated
from orbit_detector.py: visualization is a distinct concern from the
grid-generation and chaos-detection logic. `_GridChaosPlottingMixin` assumes
the host class provides `chaos_grids`, `x_grid`, `vx_grid`,
`energy_remainder`, plus `_dispatch_plot`/`_resolve_backend` (already
available via `OrbitChaosDetector`'s own `_OrbitPlottingMixin`).
"""

from typing import Any

from .visualisation import (
    plot_chaos_maps_mpl,
    plot_chaos_maps_plotly,
    plot_composite_chaos_map_mpl,
    plot_composite_chaos_map_plotly,
)


class _GridChaosPlottingMixin:
    """`plot_chaos_map`, `plot_composite_chaos_map`, and `save_chaos_maps`."""

    def plot_chaos_map(
        self,
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        show_resonances: bool = True,
        **kwargs,
    ) -> Any:
        """Plot SALI, GALI, and Lyapunov chaos maps side-by-side over the grid.

        Parameters
        ----------
        backend : str, optional
            'matplotlib' or 'plotly' (overrides the detector's default).
        save_path : str, optional
            Path to export the figure. For plotly, a '.html' path writes an
            interactive file; any other extension is passed to
            `fig.write_image` (requires the `kaleido` package).
        show : bool, default True
            Display the figure immediately.
        show_resonances : bool, default True
            Overlay corotation/Lindblad radii (see `resonance_radii`) as
            mirrored vertical lines. Set False to omit them (e.g. for a
            non-rotating potential where they'd all be absent anyway).
        """
        sali_grid, gali_grid, lyap_grid = self.chaos_grids
        return self._dispatch_plot(
            plot_chaos_maps_mpl,
            plot_chaos_maps_plotly,
            backend,
            sali_grid=sali_grid,
            gali_grid=gali_grid,
            lyapunov_grid=lyap_grid,
            x_vals=self.x_grid,
            v_x_vals=self.vx_grid,
            E_rem_vals=self.energy_remainder,
            resonance_radii=self.resonance_radii if show_resonances else None,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def plot_composite_chaos_map(
        self,
        backend: str | None = None,
        save_path: str | None = None,
        show: bool = True,
        show_resonances: bool = True,
        **kwargs,
    ) -> Any:
        """Plot a single RGB composite overlay of the SALI/GALI/Lyapunov maps.

        Parameters mirror `plot_chaos_map` — see its docstring.
        """
        sali_grid, gali_grid, lyap_grid = self.chaos_grids
        return self._dispatch_plot(
            plot_composite_chaos_map_mpl,
            plot_composite_chaos_map_plotly,
            backend,
            sali_grid=sali_grid,
            gali_grid=gali_grid,
            lyapunov_grid=lyap_grid,
            x_vals=self.x_grid,
            v_x_vals=self.vx_grid,
            E_rem_vals=self.energy_remainder,
            resonance_radii=self.resonance_radii if show_resonances else None,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def save_chaos_maps(
        self,
        side_by_side_path: str | None = None,
        composite_path: str | None = None,
        backend: str | None = None,
        **kwargs,
    ) -> None:
        """Save the side-by-side and/or composite chaos maps to disk without
        displaying them.

        Parameters
        ----------
        side_by_side_path : str, optional
            If given, renders `plot_chaos_map` and saves it here.
        composite_path : str, optional
            If given, renders `plot_composite_chaos_map` and saves it here.
        backend : str, optional
            'matplotlib' or 'plotly' (overrides the detector's default).
        """
        if side_by_side_path is not None:
            self.plot_chaos_map(
                backend=backend, save_path=side_by_side_path, show=False, **kwargs
            )
        if composite_path is not None:
            self.plot_composite_chaos_map(
                backend=backend, save_path=composite_path, show=False, **kwargs
            )
