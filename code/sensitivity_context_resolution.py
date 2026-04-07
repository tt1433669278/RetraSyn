import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / 'code'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'context_sensitivity'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = Path(sys.executable)

DATASET = 'tdrive'
METHOD = 'retrasyn_pst'
PROFILE = 'balanced'
EPSILON = '1.0'
GRID_NUM = '10'
WINDOW = '20'
PHI = '20'
CONTEXT_GRIDS = ['10', '8', '6', '4']
EXTRA_ARGS = [
    '--pst_projection_strength', '0.6',
    '--pst_projection_feasible_floor', '0.25',
    '--pst_turn_weight', '0.45',
    '--pst_accel_weight', '0.55',
]

METRIC_ORDER = [
    'Density Error',
    'Transition Error',
    'Spatial-Temporal Query Error',
    'Pattern F1 Error',
    'Kendall-tau Coefficient',
    'Length Error',
    'Trip Error',
    'Physical Violation Error',
    'Hotspot NDCG',
]

PLOT_METRICS = [
    'Transition Error',
    'Pattern F1 Error',
    'Trip Error',
    'Physical Violation Error',
    'Hotspot NDCG',
    'Kendall-tau Coefficient',
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
    syn_dir = PROJECT_ROOT / 'data' / 'syn_data' / DATASET
    aliases = []

    for ctx in CONTEXT_GRIDS:
        alias = f'retrasyn_pst_{PROFILE}_g{GRID_NUM}_ctx{ctx}'
        aliases.append(alias)
        dst_pkl = syn_dir / f'{alias}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.pkl'
        dst_meta = syn_dir / f'{alias}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.meta.json'
        if dst_pkl.exists() and dst_meta.exists():
            continue

        gen_log = OUTPUT_DIR / f'ctx{ctx}_generate.log'
        cmd = [
            str(PYTHON_EXE),
            str(CODE_DIR / 'RetraSyn_b.py'),
            '--dataset', DATASET,
            '--method', METHOD,
            '--pst_profile', PROFILE,
            '--epsilon', EPSILON,
            '--grid_num', GRID_NUM,
            '--w', WINDOW,
            '--phi', PHI,
            '--pst_context_grid_num', ctx,
            *EXTRA_ARGS,
        ]
        run_command(cmd, gen_log)

        src_pkl = syn_dir / f'retrasyn_pst_{PROFILE}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.pkl'
        src_meta = syn_dir / f'retrasyn_pst_{PROFILE}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.meta.json'
        shutil.copy2(src_pkl, dst_pkl)
        if src_meta.exists():
            shutil.copy2(src_meta, dst_meta)

    eval_log = OUTPUT_DIR / 'context_sensitivity_evaluate.log'
    eval_cmd = [
        str(PYTHON_EXE),
        str(CODE_DIR / 'evaluation_b.py'),
        '--dataset', DATASET,
        '--method', METHOD,
        '--pst_profile', PROFILE,
        '--compare_methods', ','.join(aliases),
        '--epsilon', EPSILON,
        '--grid_num', GRID_NUM,
        '--w', WINDOW,
        '--phi', PHI,
    ]
    eval_text = run_command(eval_cmd, eval_log)

    blocks = re.split(r'INFO:evaluation:===== ', eval_text)
    rows = []
    for block in blocks:
        if not block.strip():
            continue
        header, *rest = block.splitlines()
        method_name = header.replace(' =====', '').strip()
        if method_name.startswith('Comparison vs') or method_name.startswith('Warning:'):
            continue
        body = '\n'.join(rest)
        row = {'method': method_name}
        row.update(parse_metrics(body))
        meta_path = syn_dir / f'{method_name}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.meta.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            meta.pop('method', None)
            row.update(meta)
        rows.append(row)

    json_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_context_sensitivity.json'
    csv_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_context_sensitivity.csv'
    png_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_context_sensitivity.png'

    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    fieldnames = ['method', 'context_grid_num', 'context_ratio'] + METRIC_ORDER
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    rows = sorted(rows, key=lambda item: item.get('context_grid_num', 0), reverse=True)
    x = [row.get('context_grid_num', float('nan')) for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, metric_name in enumerate(PLOT_METRICS):
        ax = axes[idx]
        y = [row.get(metric_name, float('nan')) for row in rows]
        ax.plot(x, y, marker='o', linewidth=2)
        ax.set_title(metric_name)
        ax.set_xlabel('context_grid_num')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Context Resolution Sensitivity ({DATASET}, grid={GRID_NUM})')
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f'Wrote: {json_path}')
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {png_path}')


if __name__ == '__main__':
    main()
