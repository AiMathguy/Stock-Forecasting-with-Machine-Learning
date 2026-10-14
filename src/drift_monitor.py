
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Use reference quantiles to define consistent bins.
    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    # A constant reference feature has no meaningful quantile bins.
    if len(edges) < 2:
        return 0.0 if np.allclose(current, reference[0]) else np.inf

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Smoothing prevents division by zero.
    epsilon = 1e-6
    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calculate_js(reference, current, bins=10):
    """Calculate Jensen-Shannon distance using shared reference bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        # Identical constants imply no observed drift.
        if np.allclose(current, reference[0]):
            return 0.0
        edges = np.array([-np.inf, np.inf])

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    epsilon = 1e-12
    p = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    q = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

    # scipy returns the square root of Jensen-Shannon divergence.
    return float(jensenshannon(p, q, base=2))


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor feature drift with PSI and JS distance."
    )
    parser.add_argument("--reference", required=True, help="Reference/training CSV")
    parser.add_argument("--current", required=True, help="Current/inference CSV")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--js-threshold", type=float, default=0.1)
    parser.add_argument("--output", default="drift_report.csv")
    args = parser.parse_args()

    if args.bins < 2:
        parser.error("--bins must be at least 2")

    reference_df = load_data(args.reference)
    current_df = load_data(args.current)

    # Only compare shared numeric columns; exclude the prediction target.
    excluded = {"target", "target_next_close", "date", "datetime"}
    features = [
        col for col in reference_df.columns
        if col in current_df.columns
        and col.lower() not in excluded
        and pd.api.types.is_numeric_dtype(reference_df[col])
        and pd.api.types.is_numeric_dtype(current_df[col])
    ]

    if not features:
        raise ValueError("No shared numeric feature columns found.")

    results = []

    for feature in features:
        ref = pd.to_numeric(reference_df[feature], errors="coerce").to_numpy()
        cur = pd.to_numeric(current_df[feature], errors="coerce").to_numpy()

        psi = calculate_psi(ref, cur, bins=args.bins)
        js = calculate_js(ref, cur, bins=args.bins)

        drifted = (
            (np.isfinite(psi) and psi >= args.psi_threshold)
            or (np.isfinite(js) and js >= args.js_threshold)
            or np.isinf(psi)
        )

        results.append({
            "feature": feature,
            "reference_rows": int(np.isfinite(ref).sum()),
            "current_rows": int(np.isfinite(cur).sum()),
            "psi": psi,
            "js_distance": js,
            "drift_detected": bool(drifted),
        })

    report = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print("\nFeature drift report")
    print(report.to_string(index=False))
    print(f"\nReport saved to: {output_path.resolve()}")
    print(f"Features flagged: {int(report['drift_detected'].sum())}/{len(report)}")

    if report["drift_detected"].any():
        print("WARNING: Feature drift threshold exceeded.")
    else:
        print("No feature exceeded the configured thresholds.")


if __name__ == "__main__":
    main()
