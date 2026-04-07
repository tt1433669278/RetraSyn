import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_PATH = PROJECT_ROOT / 'results' / 'projection_ablation_adaptive' / 'tdrive_balanced_g6_adaptive_projection_ablation.json'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'paper_assets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RADAR_PATH = OUTPUT_DIR / 'metrics_radar_tradeoff.png'
DUAL_AXIS_PATH = OUTPUT_DIR / 'metrics_dual_axis_tradeoff.png'


def load_rows():
    rows = json.loads(ABLATION_PATH.read_text(encoding='utf-8'))
    by_name = {row['method']: row for row in rows}
    methods = [
        ('RetraSyn', by_name['retrasyn']),
        ('K-PST Off', by_name['retrasyn_pst_balanced_projoff']),
        ('K-PST Ada-0.6', by_name['retrasyn_pst_balanced_adaptive06']),
        ('K-PST Ada-1.0', by_name['retrasyn_pst_balanced_adaptive10']),
        ('K-PST Ada-1.4', by_name['retrasyn_pst_balanced_adaptive14']),
    ]
    return methods


def normalize_benefit(values, higher_is_better):
    arr = np.asarray(values, dtype=float)
    if higher_is_better:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if np.isclose(lo, hi):
            return np.ones_like(arr)
        return (arr - lo) / (hi - lo + 1e-8)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if np.isclose(lo, hi):
        return np.ones_like(arr)
    return (hi - arr) / (hi - lo + 1e-8)


def plot_radar(methods):
    metric_defs = [
        ('Pattern F1 Error', True, 'Pattern'),
        ('Hotspot NDCG', True, 'Hotspot'),
        ('Trip Error', False, 'Trip'),
        ('Physical Violation Error', False, 'Phys'),
    ]
    labels = [m[2] for m in metric_defs]
    values_by_metric = {}
    for metric_name, higher_is_better, _ in metric_defs:
        values_by_metric[metric_name] = normalize_benefit(
            [row[metric_name] for _, row in methods],
            higher_is_better
        )

    angles = np.linspace(0, 2 * np.pi, len(metric_defs), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = plt.subplot(111, polar=True)
    palette = ['#475569', '#2563eb', '#0f766e', '#f59e0b', '#dc2626']
    for idx, (label, row) in enumerate(methods):
        series = [values_by_metric[metric_name][idx] for metric_name, _, _ in metric_defs]
        series += series[:1]
        ax.plot(angles, series, linewidth=2, label=label, color=palette[idx])
        ax.fill(angles, series, alpha=0.10, color=palette[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title('Utility Trade-off Radar', pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12))
    fig.tight_layout()
    fig.savefig(RADAR_PATH, dpi=220)
    plt.close(fig)


def plot_dual_axis(methods):
    variant_methods = methods[2:]
    labels = [label for label, _ in variant_methods]
    x = np.arange(len(labels))
    hotspot = [row['Hotspot NDCG'] for _, row in variant_methods]
    phys = [row['Physical Violation Error'] for _, row in variant_methods]
    trip = [row['Trip Error'] for _, row in variant_methods]
    pattern = [row['Pattern F1 Error'] for _, row in variant_methods]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax2 = ax1.twinx()

    ax1.plot(x, hotspot, marker='o', linewidth=2.2, color='#0f766e', label='Hotspot NDCG')
    ax1.plot(x, pattern, marker='s', linewidth=2.2, color='#2563eb', label='Pattern F1')
    ax2.plot(x, phys, marker='^', linewidth=2.2, color='#dc2626', label='Physical Violation Error')
    ax2.plot(x, trip, marker='D', linewidth=2.2, color='#f59e0b', label='Trip Error')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Higher is better')
    ax2.set_ylabel('Lower is better')
    ax1.set_title('Hotspot-Pattern vs Physical-Trip Trade-off')
    ax1.grid(True, axis='y', alpha=0.3)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper center', ncol=2)
    fig.tight_layout()
    fig.savefig(DUAL_AXIS_PATH, dpi=220)
    plt.close(fig)


def main():
    methods = load_rows()
    plot_radar(methods)
    plot_dual_axis(methods)
    print(f'Wrote: {RADAR_PATH}')
    print(f'Wrote: {DUAL_AXIS_PATH}')


if __name__ == '__main__':
    main()
