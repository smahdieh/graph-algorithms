"""Short demo of the four algorithms (runs in a few seconds).

1. Small graphs where the right answer is known.
2. One BHOSLIB/FRB graph with a known optimal cover (if data/frb/ exists).
3. A timing comparison on a random graph with 100,000 vertices.

Usage:
    python demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from graph_algorithms.coloring import (  # noqa: E402
    greedy_coloring,
    is_proper_coloring,
    num_colors,
    smallest_last_coloring,
)
from graph_algorithms.generators import (  # noqa: E402
    complete,
    crown,
    cycle,
    random_sparse_graph,
    random_tree,
    star,
)
from graph_algorithms.graph import Graph, load_dimacs  # noqa: E402
from graph_algorithms.vertex_cover import (  # noqa: E402
    basic_vertex_cover,
    is_vertex_cover,
    randomized_vertex_cover,
)


def run_all(graph: Graph) -> dict[str, int]:
    """Run the four algorithms, check each output, and return its size.

    Input:  graph -- the graph to run on.
    Output: dict with the basic and randomized cover sizes ("basic", "rand")
            and the colors used by greedy and smallest-last ("greedy", "sl").
    """
    basic = basic_vertex_cover(graph)
    rand = randomized_vertex_cover(graph, seed=0)
    greedy = greedy_coloring(graph)
    smallest_last = smallest_last_coloring(graph)
    assert is_vertex_cover(graph, basic) and is_vertex_cover(graph, rand)
    assert is_proper_coloring(graph, greedy) and is_proper_coloring(graph, smallest_last)
    return {"basic": len(basic), "rand": len(rand),
            "greedy": num_colors(greedy), "sl": num_colors(smallest_last)}


def section(title: str) -> None:
    """Print a section heading."""
    print(f"\n{title}\n{'-' * len(title)}")


def demo_small_graphs() -> None:
    """Known answers: optimal cover size and minimum number of colors."""
    section("1. Small graphs with known answers")
    graphs = [
        # name, graph, optimal cover, minimum colors
        ("star, 10 leaves", star(10), 1, 2),
        ("complete K6", complete(6), 5, 6),
        ("cycle C9", cycle(9), 5, 3),
        ("random tree, 40", random_tree(40, seed=1), None, 2),
        ("crown, 2 x 8", crown(8), 8, 2),  # bipartite, perfect matching of size 8
    ]
    print(f"{'graph':18} {'basic VC':>9} {'rand. VC':>9} {'opt. VC':>8}"
          f" {'greedy col.':>12} {'SL col.':>8} {'min col.':>9}")
    for name, graph, opt, chi in graphs:
        r = run_all(graph)
        print(f"{name:18} {r['basic']:>9} {r['rand']:>9} {opt if opt else '-':>8}"
              f" {r['greedy']:>12} {r['sl']:>8} {chi:>9}")
    print("All covers are valid and all colorings are proper. On the crown graph the"
          "\nbasic greedy order needs 8 colors; smallest-last finds the optimal 2.")


def demo_frb() -> None:
    """One FRB instance, compared with its known optimal cover."""
    section("2. BHOSLIB/FRB benchmark graph")
    path = ROOT / "data" / "frb" / "frb40-19-1.mis"
    if not path.exists():
        print("Skipped: run `python data/download_dataset.py` first.")
        return
    graph = load_dimacs(path)
    opt = 40 * 19 - 40
    r = run_all(graph)
    print(f"{path.name}: {graph.num_vertices} vertices, {graph.num_edges} edges,"
          f" optimal cover {opt}")
    print(f"  basic cover      {r['basic']:5}  ({r['basic'] / opt:.3f} x optimum)")
    print(f"  randomized cover {r['rand']:5}  ({r['rand'] / opt:.3f} x optimum)")
    print(f"  greedy coloring  {r['greedy']:5} colors")
    print(f"  smallest-last    {r['sl']:5} colors")


def demo_timing() -> None:
    """Time each algorithm on one large random graph."""
    section("3. Timing on a random graph")
    graph = random_sparse_graph(100_000, avg_degree=10, seed=42)
    print(f"{graph.num_vertices} vertices, {graph.num_edges} edges")
    runs = [
        ("basic cover", lambda: len(basic_vertex_cover(graph)), "vertices"),
        ("randomized cover", lambda: len(randomized_vertex_cover(graph, seed=0)), "vertices"),
        ("greedy coloring", lambda: num_colors(greedy_coloring(graph)), "colors"),
        ("smallest-last coloring", lambda: num_colors(smallest_last_coloring(graph)), "colors"),
    ]
    for name, run, unit in runs:
        start = time.perf_counter()
        size = run()
        ms = (time.perf_counter() - start) * 1000
        print(f"  {name:24} {ms:8.1f} ms   {size} {unit}")


if __name__ == "__main__":
    start = time.perf_counter()
    demo_small_graphs()
    demo_frb()
    demo_timing()
    print(f"\nDone in {time.perf_counter() - start:.1f} s.")
