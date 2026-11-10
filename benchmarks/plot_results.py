"""Make the report figures and summary table from results/results.csv (seaborn).

Writes to results/figures/:
  runtime.pdf        runtime vs. graph size, random and FRB graphs
  cover_memory.pdf   (a) cover size / optimum on FRB, (b) peak memory on random graphs
  colors.pdf         colors used: greedy vs. smallest-last
  summary_table.tex  LaTeX table of per-family averages
  summary_table.csv  the same numbers as CSV
(.png copies of each figure are written too, for the README / quick viewing.)

The figures are drawn at the exact text width of the ACM small template and
typeset by LaTeX with the template's fonts (Linux Libertine, 10 pt), so they
match the report when included with width=\\linewidth. Use --no-latex if no
LaTeX installation is available (falls back to matplotlib's own fonts).

Usage:
  python benchmarks/plot_results.py
  python benchmarks/plot_results.py --copy-to ../report/figures
  python benchmarks/plot_results.py --csv results/results_quick.csv --no-latex
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Fixed label, colour and marker per algorithm, used in every figure. The
# colours are categorical slots 1-5, validated for colour-blind separation;
# the markers make the plots readable in greyscale too.
LABELS = {
    "basic_cover": "Basic cover",
    "randomized_cover": "Randomized cover",
    "greedy_coloring": "Greedy coloring",
    "smallest_last_coloring": "Smallest-last coloring",
    "smallest_last_order": "Smallest-last ordering only",
}
ORDER = list(LABELS.values())
PALETTE = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D", "v"]))
INK, MUTED = "#2b2b2a", "#6b6a63"
SIZE_LABEL = r"$|V| + |E|$"

TEXT_WIDTH = 395.8225 / 72.27  # \textwidth of acmsmall in inches
BODY_PT, SMALL_PT = 10, 9  # \normalsize and \small of acmsmall
LATEX_PREAMBLE = (  # the same fonts acmart loads
    r"\usepackage[T1]{fontenc}"
    r"\usepackage[tt=false, type1=true]{libertine}"
    r"\usepackage[varqu]{zi4}"
    r"\usepackage[libertine]{newtxmath}"
)
LEGEND_ROW = 0.25  # inches per legend row reserved at the top of each figure


def setup_style(use_latex: bool) -> None:
    """Seaborn theme with the report's fonts and sizes.

    Input: whether to typeset text with LaTeX. Output: None (sets rcParams).
    """
    sns.set_theme(context="paper", style="whitegrid", font="serif", rc={
        "font.size": BODY_PT, "axes.titlesize": BODY_PT, "axes.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT, "xtick.labelsize": SMALL_PT, "ytick.labelsize": SMALL_PT,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "grid.color": "#e4e3dd",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 5,
        "pdf.fonttype": 42,
    })
    if use_latex:
        plt.rcParams.update({"text.usetex": True, "text.latex.preamble": LATEX_PREAMBLE})


def load_frame(path: Path) -> pd.DataFrame:
    """Read the benchmark CSV and add plotting columns.

    Input:  path -- CSV written by run_benchmarks.py.
    Output: DataFrame with extra columns label (display name), size (V + E),
            time_ms and memory_kib.
    """
    df = pd.read_csv(path)
    df["label"] = df["algorithm"].map(LABELS)
    df["size"] = df["n"] + df["m"]
    df["time_ms"] = df["time_median_s"] * 1000
    df["memory_kib"] = df["peak_memory_bytes"] / 1024
    return df


def short_number(value: float, _pos: int | None = None) -> str:
    """Format a number compactly: 2000 -> '2k', 1500000 -> '1.5M'."""
    for limit, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def log_axes(ax: plt.Axes) -> None:
    """Log-log axes with plain tick labels (1, 2, 5, 10, ...)."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(short_number))
        axis.set_minor_formatter(NullFormatter())


def lines(ax: plt.Axes, data: pd.DataFrame, y: str, labels: list[str]) -> None:
    """Mean line per algorithm, with a min-max band over a family's instances."""
    sns.lineplot(data=data, x="size", y=y, hue="label", style="label", hue_order=labels,
                 style_order=labels, palette=PALETTE, markers=MARKERS, dashes=False,
                 errorbar=("pi", 100), err_kws={"alpha": 0.18},
                 markeredgecolor="white", markeredgewidth=0.6, ax=ax, legend=False)


def linear_reference(ax: plt.Axes, data: pd.DataFrame, y: str) -> None:
    """Dashed guide of slope 1 (linear growth), at half the lowest first point."""
    sizes = sorted(data["size"].unique())
    first = data[data["size"] == sizes[0]][y].min() / 2
    ax.plot([sizes[0], sizes[-1]], [first, first * sizes[-1] / sizes[0]],
            color=MUTED, linestyle="--", linewidth=1)


def shared_legend(fig: plt.Figure, labels: list[str], bars: bool = False,
                  reference: bool = False, ncol: int = 3) -> None:
    """One legend above the panels: colour + marker (or a swatch for bars)."""
    if bars:
        handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[lab], label=lab) for lab in labels]
    else:
        handles = [plt.Line2D([], [], color=PALETTE[lab], marker=MARKERS[lab], label=lab,
                              markeredgecolor="white", markeredgewidth=0.6) for lab in labels]
    if reference:
        handles.append(plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1,
                                  label="Linear growth"))
    fig.legend(handles=handles, loc="upper center", ncol=min(ncol, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.8,
               columnspacing=1.4, borderaxespad=0.1)


def layout(fig: plt.Figure, legend_rows: int = 1) -> None:
    """tight_layout that leaves the top strip free for the shared legend.

    Also puts the x-axis labels of side-by-side panels on one line, so the
    (a), (b) labels under them line up even when the tick labels differ.
    """
    top = 1 - (LEGEND_ROW * legend_rows + 0.05) / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top), pad=0.3)
    fig.align_xlabels()


def panel(ax: plt.Axes, letter: str) -> None:
    """Label a subplot (a), (b), ... centred below it, under the x-axis label."""
    ax.annotate(f"({letter})", xy=(0.5, 0), xycoords=ax.xaxis.label, xytext=(0, -4),
                textcoords="offset points", ha="center", va="top", fontsize=BODY_PT)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Write the figure as PDF (for LaTeX) and PNG (for quick viewing).

    No bbox_inches="tight": that would trim the page below the text width and
    LaTeX would then scale the figure (and its fonts) up to \\linewidth.
    """
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf")


def plot_runtime(df: pd.DataFrame, out_dir: Path) -> None:
    """(a) random sparse graphs, (b) FRB graphs: runtime vs. V + E, log-log."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.1))
    for ax, dataset, letter in zip(axes, ("random", "frb"), "ab"):
        data = df[df["dataset"] == dataset]
        lines(ax, data, "time_ms", ORDER)
        linear_reference(ax, data, "time_ms")
        log_axes(ax)
        ax.set(xlabel=SIZE_LABEL, ylabel="Median runtime (ms)" if letter == "a" else "")
        panel(ax, letter)
    shared_legend(fig, ORDER, reference=True)
    layout(fig, legend_rows=2)
    save(fig, out_dir, "runtime")


def plot_cover_memory(df: pd.DataFrame, out_dir: Path) -> None:
    """(a) cover size / optimum per FRB family, (b) peak memory on random graphs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.0))

    covers = [LABELS["basic_cover"], LABELS["randomized_cover"]]
    data = df[(df["dataset"] == "frb") & df["label"].isin(covers)]
    sns.lineplot(data=data, x="family", y="ratio", hue="label", style="label",
                 hue_order=covers, style_order=covers, palette=PALETTE, markers=MARKERS,
                 dashes=False, errorbar=("pi", 100), err_kws={"alpha": 0.18},
                 markeredgecolor="white", markeredgewidth=0.6, sort=False, ax=ax1, legend=False)
    ax1.set(xlabel="FRB family", ylabel="Cover size / optimum")
    ax1.tick_params(axis="x", rotation=45)
    for tick in ax1.get_xticklabels():
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")
    panel(ax1, "a")

    data = df[df["dataset"] == "random"]
    lines(ax2, data, "memory_kib", ORDER)
    linear_reference(ax2, data, "memory_kib")
    log_axes(ax2)
    ax2.set(xlabel=SIZE_LABEL, ylabel="Peak extra memory (KiB)")
    panel(ax2, "b")

    shared_legend(fig, ORDER, reference=True)
    layout(fig, legend_rows=2)
    save(fig, out_dir, "cover_memory")


def plot_colors(df: pd.DataFrame, out_dir: Path) -> None:
    """(a) FRB, (b) random graphs: colors used by greedy vs. smallest-last."""
    labels = [LABELS["greedy_coloring"], LABELS["smallest_last_coloring"]]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))
    for ax, dataset, letter in zip(axes, ("frb", "random"), "ab"):
        data = df[(df["dataset"] == dataset) & df["label"].isin(labels)].copy()
        data["group"] = data["family"] if dataset == "frb" else data["n"].map(short_number)
        sns.barplot(data=data, x="group", y="result_mean", hue="label", hue_order=labels,
                    palette=PALETTE, saturation=1, errorbar=None, gap=0.1, ax=ax, legend=False)
        ax.set(xlabel="FRB family" if dataset == "frb" else "Vertices $n$",
               ylabel="Colors used" if letter == "a" else "")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
            tick.set_rotation_mode("anchor")
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)  # keep grid lines behind the bars
        panel(ax, letter)
    shared_legend(fig, labels, bars=True)
    layout(fig)
    save(fig, out_dir, "colors")


def summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-family averages as CSV and as a LaTeX tabular (booktabs)."""
    rows = []
    for dataset in ("frb", "random"):
        data = df[df["dataset"] == dataset].copy()
        data["group"] = (data["family"] if dataset == "frb"
                         else "random $n$=" + data["n"].map(short_number))
        for group, g in data.groupby("group", sort=False):
            mean = g.groupby("algorithm").mean(numeric_only=True)
            rows.append({
                "Graph": group,
                "V": f"{mean.loc['basic_cover', 'n']:.0f}",
                "E": f"{mean.loc['basic_cover', 'm']:.0f}",
                "Basic VC": f"{mean.loc['basic_cover', 'result_mean']:.0f}",
                "Rand. VC": f"{mean.loc['randomized_cover', 'result_mean']:.0f}",
                "Opt. VC": (f"{mean.loc['basic_cover', 'optimum']:.0f}"
                            if dataset == "frb" else "--"),
                "Greedy col.": f"{mean.loc['greedy_coloring', 'result_mean']:.1f}",
                "SL col.": f"{mean.loc['smallest_last_coloring', 'result_mean']:.1f}",
                **{f"{name} ms": f"{mean.loc[algo, 'time_ms']:.1f}" for algo, name in
                   (("basic_cover", "Basic VC"), ("randomized_cover", "Rand. VC"),
                    ("greedy_coloring", "Greedy"), ("smallest_last_coloring", "SL"))},
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "summary_table.csv", index=False)
    # LaTeX: grouped header (vertex cover / colors / time) so the table fits
    # the text width; random graphs are labelled by their vertex count.
    groups = (r" & & & \multicolumn{3}{c}{Vertex cover} & \multicolumn{2}{c}{Colors}"
              r" & \multicolumn{4}{c}{Time (ms)} \\")
    rules = r"\cmidrule(lr){4-6} \cmidrule(lr){7-8} \cmidrule(lr){9-12}"
    header = (r"Graph & $|V|$ & $|E|$ & Basic & Rand. & Opt. & Greedy & SL"
              r" & Basic & Rand. & Greedy & SL \\")
    cells = out.astype(str).values.tolist()
    body = []
    for row in cells:
        if row[0].startswith("random"):
            row[0] = "random, " + row[0].split("=")[1]
        body.append(" & ".join(row) + r" \\")
    first_random = next(i for i, r in enumerate(cells) if r[0].startswith("random"))
    body.insert(first_random, r"\midrule")
    tex = [r"\begin{tabular}{l" + "r" * (len(out.columns) - 1) + "}", r"\toprule", groups,
           rules, header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    (out_dir / "summary_table.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {out_dir / 'summary_table'}.tex/.csv")
    print(out.to_string(index=False))


def main() -> None:
    """Parse options, then make every figure and the summary table."""
    parser = argparse.ArgumentParser(description="Plot the benchmark results.")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--copy-to", type=Path, help="also copy PDFs and the table here")
    parser.add_argument("--no-latex", action="store_true",
                        help="use matplotlib's fonts instead of LaTeX")
    args = parser.parse_args()

    use_latex = not args.no_latex and shutil.which("latex") is not None
    setup_style(use_latex)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(args.csv)
    plot_runtime(df, args.out_dir)
    plot_cover_memory(df, args.out_dir)
    plot_colors(df, args.out_dir)
    summary(df, args.out_dir)
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for name in ("runtime", "cover_memory", "colors"):
            shutil.copy(args.out_dir / f"{name}.pdf", args.copy_to)
        shutil.copy(args.out_dir / "summary_table.tex", args.copy_to)


if __name__ == "__main__":
    main()
