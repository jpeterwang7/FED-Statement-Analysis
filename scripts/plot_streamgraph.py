"""
Streamgraph of theme shares across the 213 scheduled FOMC statements.

Reads fomc_statement_themes_wide.csv and writes
  figures/theme_streamgraph.png    static streamgraph (matplotlib, wiggle baseline)
  figures/theme_streamgraph.html   interactive 100% stacked area with hover (plotly)

Shares are smoothed for display with a centered 3-statement rolling mean; the CSV keeps raw shares.
Run:  .venv/bin/python scripts/plot_streamgraph.py   (after theme_shares.py)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from theme_shares import THEMES

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
SMOOTH = 3

# Categorical slots in fixed order (validated adjacent-pair palette); "other" is neutral gray.
COLORS = dict(zip(THEMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                           "#008300", "#6250d6", "#e34948", "#a3a29c"]))
LABELS = {t: t.replace("_", " ").replace("energy geopolitics", "energy / geopolitics").capitalize()
          for t in THEMES}
CHAIRS = [("Greenspan", "2000-01-01", "2006-02-01"), ("Bernanke", "2006-02-01", "2014-02-03"),
          ("Yellen", "2014-02-03", "2018-02-05"), ("Powell", "2018-02-05", "2026-05-22"),
          ("Warsh", "2026-05-22", "2026-12-31")]
EVENTS = [("2008-09-16", "Lehman"), ("2020-03-15", "COVID")]
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#f0efec"


def load():
    w = pd.read_csv(ROOT / "fomc_statement_themes_wide.csv", parse_dates=["release_date"])
    w = w[(w["meeting_type"] == "scheduled") & (w["statement_kind"] == "main")]
    w = w.sort_values("release_date").set_index("release_date")[THEMES]
    w = w.loc[:, w.sum() > 0]  # a theme with no sentences gets no band or legend entry
    smooth = w.rolling(SMOOTH, center=True, min_periods=1).mean()
    return w, smooth.div(smooth.sum(axis=1), axis=0)


def png(s):
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (name, a, b) in enumerate(CHAIRS):
        a, b = pd.Timestamp(a), min(pd.Timestamp(b), s.index.max() + pd.Timedelta(days=60))
        if i % 2:
            ax.axvspan(a, b, color=BAND, zorder=0, lw=0)
        ax.text(a + (b - a) / 2, 1.012, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.stackplot(s.index, *[s[t] for t in s.columns], labels=[LABELS[t] for t in s.columns],
                 colors=[COLORS[t] for t in s.columns], baseline="wiggle",
                 edgecolor="#fcfcfb", linewidth=0.6, zorder=2)
    for d, label in EVENTS:
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=3)
        ax.text(pd.Timestamp(d), 0.01, f" {label}", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=MUTED, va="bottom")
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(s.index.min(), s.index.max())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("What FOMC statements talk about, 2000–2026", x=0.06, ha="left",
                 fontsize=14, color=INK, y=0.98)
    ax.set_title("Share of each scheduled statement's sentences by theme "
                 f"(admin text excluded; {SMOOTH}-statement rolling mean). Width = 100%.",
                 loc="left", fontsize=9.5, color=MUTED, pad=30)
    fig.tight_layout()
    fig.savefig(FIG / "theme_streamgraph.png", facecolor=fig.get_facecolor())


def html(raw, s):
    fig = go.Figure()
    for t in s.columns:
        fig.add_trace(go.Scatter(
            x=s.index, y=s[t] * 100, name=LABELS[t], mode="lines", stackgroup="one",
            line=dict(width=0.5, color="#fcfcfb"), fillcolor=COLORS[t],
            customdata=raw[t] * 100,
            hovertemplate=f"{LABELS[t]}: %{{customdata:.0f}}% (smoothed %{{y:.0f}}%)<extra></extra>"))
    for name, a, b in CHAIRS[1::2]:
        fig.add_vrect(x0=a, x1=min(pd.Timestamp(b), s.index.max()), fillcolor=BAND,
                      line_width=0, layer="below")
    for name, a, b in CHAIRS:
        a = max(pd.Timestamp(a), s.index.min())
        fig.add_annotation(x=a + (min(pd.Timestamp(b), s.index.max()) - a) / 2, y=1.05,
                           yref="paper", text=name, showarrow=False,
                           font=dict(size=11, color=MUTED))
    for d, label in EVENTS:
        fig.add_vline(x=pd.Timestamp(d).timestamp() * 1000, line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text=label, annotation_position="bottom right",
                      annotation_font=dict(size=10, color=MUTED))
    fig.update_layout(
        title=dict(text="What FOMC statements talk about, 2000–2026<br><sup>Share of each scheduled "
                        f"statement's sentences by theme; admin text excluded; {SMOOTH}-statement rolling "
                        "mean. Hover shows the raw share.</sup>", x=0.01),
        hovermode="x unified", plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        font=dict(color=INK), legend=dict(traceorder="reversed"),
        yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#e8e7e3", title=None),
        xaxis=dict(showgrid=False), margin=dict(t=110))
    fig.write_html(FIG / "theme_streamgraph.html", include_plotlyjs="cdn")


def main():
    FIG.mkdir(exist_ok=True)
    raw, s = load()
    png(s)
    html(raw, s)
    print(f"{len(s)} scheduled statements plotted to {FIG.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
