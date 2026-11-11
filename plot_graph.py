"""Draw the synthetic hospital graph. Run: python plot_graph.py -> figures/hospital_graph.png"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import ALT, COMMIT_EDGE, ED, EDGES, J1, J2, J3, J4, SUPPLY

XY = {SUPPLY: (0, 0), J1: (1, 0), J2: (2, 0), J3: (3, 0), J4: (6, 0), ED: (7, 0), ALT: (4, -1.6)}


def main():
    fig, ax = plt.subplots(figsize=(8, 3))
    for (a, b), L in EDGES.items():
        (x1, y1), (x2, y2) = XY[a], XY[b]
        commit = (a, b) == COMMIT_EDGE
        ax.plot([x1, x2], [y1, y2], color="#c0392b" if commit else "#555",
                lw=4 if commit else 1.5, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, f"{L}", ha="center", fontsize=9)
    for n, (x, y) in XY.items():
        color = {SUPPLY: "#2e86c1", ED: "#27ae60", ALT: "#f39c12"}.get(n, "#ddd")
        ax.scatter(x, y, s=700, color=color, edgecolor="k", zorder=2)
        ax.text(x, y, n, ha="center", va="center", fontsize=6 if len(n) > 3 else 8, zorder=3)
    ax.text(4.5, 0.35, "commitment segment: c_rev when reversing J4->J3",
            color="#c0392b", ha="center", fontsize=8)
    ax.set_title("Synthetic hospital graph (edge labels = travel steps; not a real layout)", fontsize=9)
    ax.axis("off")
    ax.set_ylim(-2.1, 0.7)
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "hospital_graph.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out / 'hospital_graph.png'}")


if __name__ == "__main__":
    main()
