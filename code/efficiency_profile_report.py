import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYN_DIR = PROJECT_ROOT / 'data' / 'syn_data' / 'tdrive'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'efficiency'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / 'tdrive_efficiency_profile.csv'
PNG_PATH = OUTPUT_DIR / 'tdrive_efficiency_profile.png'
TIMESTAMPS = 886

TARGETS = [
    ('Proj-Off g6', 'retrasyn_pst_balanced_projoff_1.0_g6_w20.meta.json'),
    ('Adaptive-0.6 g6', 'retrasyn_pst_balanced_adaptive06_1.0_g6_w20.meta.json'),
    ('Adaptive-1.0 g6', 'retrasyn_pst_balanced_adaptive10_1.0_g6_w20.meta.json'),
    ('Adaptive-1.4 g6', 'retrasyn_pst_balanced_adaptive14_1.0_g6_w20.meta.json'),
]


def main():
    rows = []
    for label, filename in TARGETS:
        meta_path = SYN_DIR / filename
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        runtime = meta.get('runtime', {})
        row = {
            'label': label,
            'spatial_decomposition_ms': 1000.0 * runtime.get('spatial_decomposition', 0.0) / TIMESTAMPS,
            'pst_update_ms': 1000.0 * runtime.get('pst_update', 0.0) / TIMESTAMPS,
            'trajectory_synthesis_ms': 1000.0 * runtime.get('trajectory_synthesis', 0.0) / TIMESTAMPS,
            'write_file_ms': 1000.0 * runtime.get('write_file', 0.0) / TIMESTAMPS,
            'projection_strength': meta.get('projection_strength', 0.0),
            'physical_violation_reduction': meta.get('physical_violation_reduction', 0.0),
        }
        rows.append(row)

    with CSV_PATH.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'label',
                'projection_strength',
                'physical_violation_reduction',
                'spatial_decomposition_ms',
                'pst_update_ms',
                'trajectory_synthesis_ms',
                'write_file_ms',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = [row['label'] for row in rows]
    x = np.arange(len(labels))
    width = 0.58

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bottom = np.zeros(len(labels))
    parts = [
        ('spatial_decomposition_ms', '#94a3b8'),
        ('pst_update_ms', '#2563eb'),
        ('trajectory_synthesis_ms', '#0f766e'),
        ('write_file_ms', '#dc2626'),
    ]
    for key, color in parts:
        values = np.asarray([row[key] for row in rows], dtype=float)
        ax.bar(x, values, width, bottom=bottom, label=key.replace('_ms', ''), color=color)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Average ms per timestamp')
    ax.set_title('Per-Timestamp Efficiency Breakdown')
    ax.legend(ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=220)
    plt.close(fig)

    print(f'Wrote: {CSV_PATH}')
    print(f'Wrote: {PNG_PATH}')


if __name__ == '__main__':
    main()
