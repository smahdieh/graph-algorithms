"""Benchmark the vertex cover and coloring algorithms.

Runs all four algorithms on
  * the BHOSLIB/FRB graphs in data/frb/ (known optimal cover size), and
  * random sparse graphs of growing size (to see how runtime scales),
and writes one CSV row per (graph, algorithm) to results/results.csv.

Usage:
    python benchmarks/run_benchmarks.py           # full run
    python benchmarks/run_benchmarks.py --quick   # small, fast run (~seconds)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_algorithms.coloring import (  # noqa: E402
    greedy_coloring,
    is_proper_coloring,
    num_colors,
    smallest_last_coloring,
    smallest_last_order,
)
from graph_algorithms.generators import random_sparse_graph  # noqa: E402
from graph_algorithms.graph import Graph, load_dimacs  # noqa: E402
from graph_algorithms.vertex_cover import (  # noqa: E402
    basic_vertex_cover,
    is_vertex_cover,
    randomized_vertex_cover,
)

FRB_DIR = ROOT / "data" / "frb"
RESULTS_DIR = ROOT / "results"

CSV_FIELDS = [
    "dataset", "graph", "family", "n", "m", "problem", "algorithm",
    "time_median_s", "time_min_s", "peak_memory_bytes",
    "result_mean", "result_min", "result_max", "optimum", "ratio",
]

# Each algorithm: (problem, function(graph, seed) -> result size, validity check)
ALGORITHMS: dict[str, tuple[str, Callable[[Graph, int], object]]] = {
    "basic_cover": ("vertex_cover", lambda g, s: basic_vertex_cover(g)),
    "randomized_cover": ("vertex_cover", lambda g, s: randomized_vertex_cover(g, seed=s)),
    "greedy_coloring": ("coloring", lambda g, s: greedy_coloring(g)),
    "smallest_last_coloring": ("coloring", lambda g, s: smallest_last_coloring(g)),
    # The ordering alone, to measure its cost separately from the coloring.
    "smallest_last_order": ("ordering", lambda g, s: smallest_last_order(g)),
}


def result_size(problem: str, graph: Graph, output: object) -> int | None:
    """Check an algorithm's output and return its size.

    Input:  problem -- "vertex_cover", "coloring" or "ordering";
            graph -- the input graph; output -- what the algorithm returned.
    Output: cover size or number of colors (None for an ordering).
    Raises AssertionError if the output is not a valid cover/coloring.
    """
    if problem == "vertex_cover":
        assert is_vertex_cover(graph, output), "invalid vertex cover"
        return len(output)
    if problem == "coloring":
        assert is_proper_coloring(graph, output), "invalid coloring"
        return num_colors(output)
    assert sorted(output) == list(graph.vertices()), "ordering is not a permutation"
    return None


def measure(graph: Graph, name: str, repeats: int, seeds: int) -> dict:
    """Time one algorithm on one graph and record the quality of its output.

    Input:  graph -- input graph; name -- key in ALGORITHMS;
            repeats -- timed runs for deterministic algorithms;
            seeds -- number of seeds (= timed runs) for the randomized one.
    Output: dict with time_median_s, time_min_s, peak_memory_bytes and
            result_mean / result_min / result_max.
    Memory is measured in one extra run with tracemalloc, so tracing does
    not slow down the timed runs.
    """
    problem, run = ALGORITHMS[name]
    run_seeds = list(range(seeds)) if name == "randomized_cover" else [0] * repeats
    times, sizes = [], []
    for seed in run_seeds:
        start = time.perf_counter()
        output = run(graph, seed)
        times.append(time.perf_counter() - start)
        sizes.append(result_size(problem, graph, output))

    tracemalloc.start()
    run(graph, 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row = {
        "time_median_s": statistics.median(times),
        "time_min_s": min(times),
        "peak_memory_bytes": peak,
    }
    if sizes[0] is not None:
        row.update(result_mean=statistics.mean(sizes), result_min=min(sizes), result_max=max(sizes))
    return row


def frb_graphs(limit_families: int | None) -> list[tuple[str, str, Graph, int]]:
    """Load the FRB graphs from data/frb/.

    Input:  limit_families -- use only the first few families (None = all).
    Output: list of (graph name, family, graph, optimal cover size).
    """
    files = sorted(FRB_DIR.glob("frb*.mis"))
    if not files:
        raise SystemExit("No FRB files found. Run: python data/download_dataset.py")
    families = sorted({f.stem.rsplit("-", 1)[0] for f in files})
    if limit_families:
        families = families[:limit_families]
    graphs = []
    for f in files:
        family = f.stem.rsplit("-", 1)[0]          # e.g. "frb30-15"
        if family not in families:
            continue
        n, k = (int(x) for x in family[3:].split("-"))
        graphs.append((f.stem, family, load_dimacs(f), n * k - n))
    return graphs


def run_all(args: argparse.Namespace) -> list[dict]:
    """Run every algorithm on every graph and return the CSV rows."""
    workloads = []
    for name, family, graph, opt in frb_graphs(2 if args.quick else None):
        workloads.append(("frb", name, family, graph, opt))
    for n in args.sizes:
        graph = random_sparse_graph(n, args.avg_degree, seed=n)
        workloads.append(("random", f"random-{n}", f"deg{args.avg_degree:g}", graph, None))

    rows = []
    for dataset, name, family, graph, opt in workloads:
        print(f"{name:16} n={graph.num_vertices:>7} m={graph.num_edges:>8}", flush=True)
        for algo, (problem, _) in ALGORITHMS.items():
            stats = measure(graph, algo, args.repeats, args.seeds)
            row = {"dataset": dataset, "graph": name, "family": family,
                   "n": graph.num_vertices, "m": graph.num_edges,
                   "problem": problem, "algorithm": algo, **stats}
            if problem == "vertex_cover" and opt is not None:
                row["optimum"] = opt
                row["ratio"] = stats["result_mean"] / opt
            rows.append(row)
            result = f"{stats['result_mean']:.1f}" if "result_mean" in stats else "-"
            print(f"    {algo:24} {stats['time_median_s'] * 1000:9.2f} ms  result={result}")
    return rows


def write_outputs(rows: list[dict], out_path: Path, args: argparse.Namespace) -> None:
    """Write the CSV rows and a small environment description."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    env = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "repeats": args.repeats,
        "seeds": args.seeds,
        "random_sizes": args.sizes,
        "avg_degree": args.avg_degree,
    }
    (out_path.parent / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print(f"\nWrote {len(rows)} rows to {out_path}")


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="2 FRB families, small random graphs, 1 repeat")
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed runs per deterministic algorithm (default 5)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="seeds for the randomized cover (default 10)")
    parser.add_argument("--avg-degree", type=float, default=10.0,
                        help="average degree of the random graphs (default 10)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "results.csv")
    args = parser.parse_args()
    if args.quick:
        args.repeats, args.seeds = 1, 3
        args.sizes = [1000, 4000]
        if args.out == RESULTS_DIR / "results.csv":
            args.out = RESULTS_DIR / "results_quick.csv"
    else:
        args.sizes = [1000 * 2 ** i for i in range(8)]  # 1k .. 128k vertices
    return args


if __name__ == "__main__":
    arguments = parse_args()
    write_outputs(run_all(arguments), arguments.out, arguments)
