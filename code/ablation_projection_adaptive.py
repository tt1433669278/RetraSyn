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
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'projection_ablation_adaptive'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = Path(sys.executable)

DATASET = 'tdrive'
METHOD = 'retrasyn_pst'
PROFILE = 'balanced'
EPSILON = '1.0'
GRID_NUM = '6'
WINDOW = '20'
PHI = '20'

GENTLE_ARGS = [
    '--pst_projection_feasible_floor', '0.25',
    '--pst_turn_weight', '0.45',
    '--pst_accel_weight', '0.55',
]

VARIANTS = [
    {'label': 'projoff', 'strength': '0.0'},
    {'label': 'adaptive06', 'strength': '0.6'},
    {'label': 'adaptive10', 'strength': '1.0'},
    {'label': 'adaptive14', 'strength': '1.4'},
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
    'Trip Error',
    'Pattern F1 Error',
    'Hotspot NDCG',
    'Physical Violation Error',
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
    rows = []
    compare_methods = ['retrasyn']

    for variant in VARIANTS:
        label = variant['label']
        strength = variant['strength']
        method_alias = f'retrasyn_pst_{PROFILE}_{label}'
        compare_methods.append(method_alias)

        gen_log = OUTPUT_DIR / f'{label}_generate.log'
        syn_dir = PROJECT_ROOT / 'data' / 'syn_data' / DATASET
        dst_pkl = syn_dir / f'{method_alias}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.pkl'
        dst_meta = syn_dir / f'{method_alias}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.meta.json'
        if dst_pkl.exists() and dst_meta.exists():
            continue

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
            '--pst_projection_strength', strength,
            *GENTLE_ARGS,
        ]
        run_command(cmd, gen_log)

        src_pkl = syn_dir / f'retrasyn_pst_{PROFILE}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.pkl'
        shutil.copy2(src_pkl, dst_pkl)
        meta_src = syn_dir / f'retrasyn_pst_{PROFILE}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.meta.json'
        if meta_src.exists():
            shutil.copy2(meta_src, dst_meta)

    eval_log = OUTPUT_DIR / 'adaptive_projection_evaluate.log'
    eval_cmd = [
        str(PYTHON_EXE),
        str(CODE_DIR / 'evaluation_b.py'),
        '--dataset', DATASET,
        '--method', METHOD,
        '--pst_profile', PROFILE,
        '--compare_methods', ','.join(compare_methods),
        '--epsilon', EPSILON,
        '--grid_num', GRID_NUM,
        '--w', WINDOW,
        '--phi', PHI,
    ]
    eval_text = run_command(eval_cmd, eval_log)

    records = []
    blocks = re.split(r'INFO:evaluation:===== ', eval_text)
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
        meta_path = PROJECT_ROOT / 'data' / 'syn_data' / DATASET / f'{method_name}_{EPSILON}_g{GRID_NUM}_w{WINDOW}.meta.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            meta.pop('method', None)
            row.update(meta)
        records.append(row)

    json_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_adaptive_projection_ablation.json'
    csv_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_adaptive_projection_ablation.csv'
    png_path = OUTPUT_DIR / f'{DATASET}_{PROFILE}_g{GRID_NUM}_adaptive_projection_ablation.png'

    json_path.write_text(json.dumps(records, indent=2), encoding='utf-8')

    extra_keys = [
        'projection_strength',
        'projection_rejection_mass',
        'feasible_mass_ratio',
        'physical_violation_reduction',
        'adaptive_projection_strength',
        'adaptive_projection_floor',
        'candidate_hotspot_density',
    ]
    fieldnames = ['method'] + extra_keys + METRIC_ORDER
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    variant_rows = [row for row in records if row['method'] != 'retrasyn']
    baseline = next((row for row in records if row['method'] == 'retrasyn'), None)
    x = [row.get('projection_strength', float('nan')) for row in variant_rows]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, metric_name in enumerate(PLOT_METRICS):
        ax = axes[idx]
        y = [row.get(metric_name, float('nan')) for row in variant_rows]
        ax.plot(x, y, marker='o', linewidth=2, label='K-PST')
        if baseline is not None:
            ax.axhline(baseline.get(metric_name, float('nan')), color='gray', linestyle='--', linewidth=1.5, label='RetraSyn')
        ax.set_title(metric_name)
        ax.set_xlabel('projection_strength')
        ax.grid(True, alpha=0.3)
    axes[-1].plot(
        [row.get('projection_strength', float('nan')) for row in variant_rows],
        [row.get('Physical Violation Error', float('nan')) for row in variant_rows],
        marker='o',
        linewidth=2,
        label='Physical Violation Error'
    )
    axes[-1].plot(
        [row.get('projection_strength', float('nan')) for row in variant_rows],
        [row.get('Hotspot NDCG', float('nan')) for row in variant_rows],
        marker='s',
        linewidth=2,
        label='Hotspot NDCG'
    )
    axes[-1].set_title('Violation vs Hotspot')
    axes[-1].set_xlabel('projection_strength')
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend()
    fig.suptitle(f'Adaptive Projection Ablation ({DATASET}, {PROFILE}, grid={GRID_NUM})')
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f'Wrote: {json_path}')
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {png_path}')


if __name__ == '__main__':
    main()
