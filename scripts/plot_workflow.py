"""Draw the AIS evaluation workflow. Run this file directly in PyCharm."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw(outputs):
    # The canvas uses a fixed grid at the intended two-column print width.
    width, height = 514, 237
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(width / 72, 2.68), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.axis("off")

    def label(x, y, text, *, title=False, align="left"):
        ax.text(x, y, text, ha=align, va="center", color="#161616",
                fontsize=10 if title else 8.2, weight="bold" if title else "normal")

    def frame(x, y, w, h, heading):
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor="#686868", linewidth=.65))
        label(x + 10, y + 13, heading, title=True)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops={
            "arrowstyle": "->", "color": "#404040", "lw": .85,
            "shrinkA": .5, "shrinkB": 1, "mutation_scale": 8,
        })

    frame(2, 2, 233, 59, "Prepare AIS tracks")
    label(12, 30, "Guam and Los Angeles: time, position, SOG, COG")
    label(12, 46, "Screen both regions; split Guam by vessel.")

    frame(279, 2, 233, 59, "Construct scenarios and windows")
    label(289, 30, "Nominal and synthetic tracks; 10 reports, stride 5")
    label(289, 46, "Positive label: intervention overlap ≥ 10%")
    arrow((235, 31), (279, 31))

    frame(2, 87, 233, 125, "Develop models in Guam")
    label(12, 116, "Use vessel-disjoint training and validation partitions.")
    label(12, 132, "Train on Generator A and nominal windows.")
    label(12, 148, "Five tabular models (44 features) and one GRU")
    ax.plot([12, 225], [159, 159], color=".8", lw=.55)
    label(12, 173, "Calibrate on label-zero validation windows from")
    label(12, 187, "nominal tracks and both generators. Freeze thresholds")
    label(12, 201, "at 1%, 3% and 5% target false-alarm rates.")

    frame(279, 87, 233, 125, "Test frozen models and thresholds")
    label(289, 116, "Test set")
    label(406, 116, "Generator A", align="center")
    label(474, 116, "Generator B", align="center")
    ax.plot([289, 501], [125, 125], color=".65", lw=.55)
    label(289, 137, "Guam test")
    label(406, 137, "A", align="center")
    label(474, 137, "C", align="center")
    label(289, 153, "Los Angeles")
    label(406, 153, "B", align="center")
    label(474, 153, "D", align="center")
    ax.plot([289, 501], [164, 164], color=".8", lw=.55)
    label(289, 179, "Stress tests: difficult normals, prevalence and")
    label(289, 195, "within-family amplitude and duration.")

    # Training and test windows follow distinct paths into the lower row.
    ax.plot([395, 395, 118], [61, 74, 74], color="#404040", lw=.85)
    arrow((118, 74), (118, 87))
    arrow((395, 74), (395, 87))
    arrow((235, 149), (279, 149))

    label(width / 2, 227,
          "Readouts: window F1 and AUPRC  ·  event recall and delay  ·  false alarms  ·  vessel-cluster intervals",
          align="center")
    for output in outputs:
        if output.suffix.lower() not in {".pdf", ".svg", ".png", ".tiff"}:
            raise ValueError("Select PDF, SVG, PNG or TIFF.")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=600, facecolor="white")
        print(output.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="*", type=Path,
                        help="Optional output filenames, including the extension.")
    paths = parser.parse_args().output
    if not paths:
        folder = Path(__file__).resolve().parents[1] / "figures"
        paths = [folder / ("Fig1_overall_workflow" + ext) for ext in (".pdf", ".svg", ".png")]
    draw(paths)
