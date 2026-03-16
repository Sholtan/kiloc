import yaml
import json
import subprocess
import itertools
from pathlib import Path
import argparse
from datetime import datetime

def main(grid_config_path):
    with open(grid_config_path) as f:
        grid_cfg = yaml.safe_load(f)

    with open(grid_cfg['base_config']) as f:
        base_cfg = yaml.safe_load(f)
    
    grid = grid_cfg.pop('grid')
    grid_cfg.pop('base_config')
    base_cfg.update(grid_cfg)

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"Grid search: {len(combinations)} runs")
    print(f"Parameters: {param_names}")

    results = []
    grid_runs_dir = Path('configs/grid_runs')

    grid_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gridrun_name = f"gridrun_{grid_timestamp}"

    for i, combo in enumerate(combinations):
        run_cfg = base_cfg.copy()
        for name, value in zip(param_names, combo):
            run_cfg[name] = value
        
        run_name = "_".join(f"{k}_{v}" for k, v in zip(param_names, combo))
        full_run_name = f"{gridrun_name}/{run_name}"

        temp_config_path = grid_runs_dir / f'run_{i:03d}.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(run_cfg, f)
        
        print(f"\nRun {i+1}/{len(combinations)} | {dict(zip(param_names, combo))}")
        
        run_dir = None
        process = subprocess.Popen(
            ['python', 'scripts/run_train.py', '--config', str(temp_config_path), '--run_name', full_run_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            print(line, end='')
            if line.startswith('RUN_DIR:'):
                run_dir = Path(line.strip().split(':', 1)[1])
        process.wait()
        if run_dir is None:
            print("WARNING: could not find RUN_DIR in output, skipping")
            continue

        with open(run_dir / 'history.json') as f:
            history = json.load(f)

        best = max(history, key=lambda x: x['f1'])
        results.append({
            'params': dict(zip(param_names, combo)),
            'run_dir': str(run_dir),
            'best_epoch': best['epoch'],
            'f1': best['f1'],
            'precision': best['precision'],
            'recall': best['recall'],
        })
        print(f"Best F1 this run: {best['f1']:.4f} at epoch {best['epoch']}")

    results.sort(key=lambda x: x['f1'], reverse=True)

    print("\n=== GRID SEARCH RESULTS ===")
    for r in results:
        print(f"F1={r['f1']:.4f} P={r['precision']:.3f} R={r['recall']:.3f} | epoch={r['best_epoch']} | {r['params']}")

    output_path = Path('checkpoints') / gridrun_name / 'grid_results.json'
    with open(output_path, 'w') as f:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_config', default='configs/grid_1.yaml')
    args = parser.parse_args()
    main(args.grid_config)