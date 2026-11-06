"""Compatibility alias for Energy Insights view."""
from app.views.energy_insights import (
    render_energy_insights,
    render_reproducibility,
)

__all__ = ["render_energy_insights", "render_reproducibility"]
