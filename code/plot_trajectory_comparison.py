import pickle
import random
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = 'tdrive'
EPSILON = '1.0'
GRID_NUM = '6'
WINDOW = '20'
LEFT_METHOD = 'retrasyn'
RIGHT_METHOD = 'retrasyn_pst_balanced_adaptive06'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'paper_assets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / 'trajectory_comparison_retrasyn_vs_kpst.png'


def load_db(method_name: str):
    path = PROJECT_ROOT / 'data' / 'syn_data' / DATASET / f'{method_name}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def select_trajectories(db, sample_size=180, min_len=6, seed=2026):
    candidates = [traj for traj in db if len(traj) >= min_len]
    random.Random(seed).shuffle(candidates)
    if len(candidates) >= sample_size:
        return candidates[:sample_size]
    return candidates


def plot_db(ax, db, title, bounds):
    min_x, min_y, max_x, max_y = bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    for traj in db:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color='#0f172a', alpha=0.08, linewidth=0.8)

    ax.scatter(
        [traj[0][0] for traj in db if traj],
        [traj[0][1] for traj in db if traj],
        s=3,
        color='#2563eb',
        alpha=0.25,
    )
    ax.scatter(
        [traj[-1][0] for traj in db if traj],
        [traj[-1][1] for traj in db if traj],
        s=3,
        color='#dc2626',
        alpha=0.25,
    )


def main():
    stats = json.loads((PROJECT_ROOT / 'data' / f'{DATASET}_stats.json').read_text(encoding='utf-8'))
    bounds = (stats['min_x'], stats['min_y'], stats['max_x'], stats['max_y'])

    left_db = select_trajectories(load_db(LEFT_METHOD), seed=2026)
    right_db = select_trajectories(load_db(RIGHT_METHOD), seed=2026)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_db(axes[0], left_db, 'RetraSyn', bounds)
    plot_db(axes[1], right_db, 'K-PST (Adaptive Projection)', bounds)

    fig.suptitle('Synthetic Trajectory Comparison on T-Drive', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)

    print(f'Wrote: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
