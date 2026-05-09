"""BERTopic interactive HTML visualizations (Tudor, Step T3.5)."""

from __future__ import annotations

from pathlib import Path


def save_default_visualizations(topic_model, out_dir: Path | str) -> dict[str, Path]:
    """Save the three required visualizations as HTML files.

    Returns a dict {viz_name: path_on_disk}. Visualizations that fail to
    generate (e.g. too few topics for a heatmap) are skipped with a warning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    plans = [
        ("topics_2d", topic_model.visualize_topics),
        ("barchart", topic_model.visualize_barchart),
        ("heatmap", topic_model.visualize_heatmap),
    ]

    for name, factory in plans:
        out_path = out_dir / f"{name}.html"
        try:
            fig = factory()
            fig.write_html(str(out_path))
            saved[name] = out_path
            print(f"saved {name:<10s} -> {out_path}")
        except (ValueError, RuntimeError) as exc:  # pragma: no cover
            print(f"skipped {name}: {exc}")

    return saved
