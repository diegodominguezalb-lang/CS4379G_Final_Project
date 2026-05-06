"""Shared formatting helpers for the NOAA Storm Events Dashboard."""

import numpy as np

# ── Colorblind-safe palette (Okabe-Ito, Lecture 21 § HCI for Viz) ────────────
# Color is NEVER the only signal; charts pair color with text labels / position.
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
]


def _fmt_usd(v: float) -> str:
    """Return a compact, human-readable USD string: $1.2T / $500B / $25M / $1.5K."""
    av = abs(v)
    if av >= 1e12: return f"${v/1e12:.2f}T"
    if av >= 1e9:  return f"${v/1e9:.2f}B"
    if av >= 1e6:  return f"${v/1e6:.2f}M"
    if av >= 1e3:  return f"${v/1e3:.2f}K"
    return f"${v:,.0f}"


def _dollar_yaxis(fig, max_val: float) -> None:
    """Replace Plotly's SI-prefix y-axis ticks (which use G for billion)
    with proper financial labels: $K / $M / $B / $T."""
    if not max_val or max_val <= 0 or not np.isfinite(max_val):
        return
    if max_val >= 1e12:
        unit, suffix = 1e12, "T"
    elif max_val >= 1e9:
        unit, suffix = 1e9, "B"
    elif max_val >= 1e6:
        unit, suffix = 1e6, "M"
    else:
        unit, suffix = 1e3, "K"
    raw = max_val / unit / 5
    mag = 10 ** int(np.floor(np.log10(max(raw, 1e-10))))
    step = np.ceil(raw / mag) * mag * unit
    vals = np.arange(0, max_val * 1.2, step)
    fig.update_yaxes(
        tickvals=vals.tolist(),
        ticktext=[f"${v/unit:.0f}{suffix}" for v in vals],
    )
