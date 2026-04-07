import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / 'code'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'budget_ablation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = Path(sys.executable)

DATASET = 'tdrive'
METHOD = 'retrasyn_pst'
PROFILE = 'balanced'
EPSILON = '1.0'
GRID_NUM = '6'
WINDOW = '20'
PHI = '20'
LENGTH_EPS_FRACS = ['0.02', '0.05', '0.10', '0.15']
METRIC_ORDER = [
    'Density Error',
    'Transition Error',
    'Spatial-Temporal Query Error',
    'Pattern F1 Error',
    'Kendall-tau Coefficient',
    'Length Error',
    'Trip Error',
    'Hotspot NDCG',
]
PLOT_METRICS = [
    'Density Error',
    'Transition Error',
    'Spatial-Temporal Query Error',
    'Pattern F1 Error',
    'Length Error',
    'Hotspot NDCG',
]


def run_command(command, log_path: Path):
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    log_path.write_text(proc.stdout + '\n' + proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed: {" ".join(command)}\nSee log: {log_path}')
    return proc.stdout + '\n' + proc.stderr


def parse_metrics(log_text: str):
    metrics = {}
    for metric_name in METRIC_ORDER:
        pattern = re.compile(rf'{re.escape(metric_name)}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
        matches = pattern.findall(log_text)
        if matches:
            metrics[metric_name] = float(matches[-1])
    return metrics


def main():
    all_rows = []

    for length_eps_frac in LENGTH_EPS_FRACS:
        tag = f'g{GRID_NUM}_len{length_eps_frac.replace(".", "p")}'
        gen_log = OUTPUT_DIR / f'{tag}_generate.log'
        eval_log = OUTPUT_DIR / f'{tag}_evaluate.log'

        gen_cmd = [
            str(PYTHON_EXE),
            str(CODE_DIR / 'RetraSyn_b.py'),
            '--dataset', DATASET,
            '--method', METHOD,
            '--pst_profile', PROFILE,
            '--epsilon', EPSILON,
            '--grid_num', GRID_NUM,
            '--w', WINDOW,
            '--phi', PHI,
            '--pst_length_eps_frac', length_eps_frac,
        ]
        run_command(gen_cmd, gen_log)

        eval_cmd = [
            str(PYTHON_EXE),
            str(CODE_DIR / 'evaluation_b.py'),
            '--dataset', DATASET,
            '--method', METHOD,
            '--pst_profile', PROFILE,
            '--compare_methods', f'retrasyn,retrasyn_pst_{PROFILE}',
            '--epsilon', EPSILON,
            '--grid_num', GRID_NUM,
            '--w', WINDOW,
            '--phi', PHI,
            '--pst_length_eps_frac', length_eps_frac,
        ]
        eval_text = run_command(eval_cmd, eval_log)
        row = {'pst_length_eps_frac': float(length_eps_frac)}
        row.update(parse_metrics(eval_text))
        all_rows.append(row)

    json_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_budget_ablation.json'
    csv_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_budget_ablation.csv'
    png_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_budget_ablation.png'

    json_path.write_text(json.dumps(all_rows, indent=2), encoding='utf-8')

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['pst_length_eps_frac'] + METRIC_ORDER)
        writer.writeheader()
        writer.writerows(all_rows)

    x = [row['pst_length_eps_frac'] for row in all_rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, metric_name in enumerate(PLOT_METRICS):
        ax = axes[idx]
        y = [row.get(metric_name, float('nan')) for row in all_rows]
        ax.plot(x, y, marker='o', linewidth=2)
        ax.set_title(metric_name)
        ax.set_xlabel('length_eps_frac')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Budget Ablation ({DATASET}, {PROFILE}, grid={GRID_NUM})')
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f'Wrote: {json_path}')
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {png_path}')


if __name__ == '__main__':
    main()
