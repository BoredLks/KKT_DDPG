# -*- coding: utf-8 -*-
"""
demo_all_sweeps.py
==================
精简示意图生成脚本: 每个扫描跑 5 个点, 每点 1 次 MC, 只 KKT。
用于确认 8 个扫描的 3 张图都能生成,数值趋势合理。

对应论文级跑法 (你本地跑):
    python sweeps.py --sweep ALL --mc-runs 500 --episodes 500 --eval-runs 3
"""
import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweeps import SWEEP_CONFIGS, run_one_point, plot_sweep

SAVE_DIR = "Figure"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)


def main():
    # 每个扫描取 5 点
    total_start = time.time()
    for sw_name, sw_cfg in SWEEP_CONFIGS.items():
        xs_full = sw_cfg['x_values']
        # 取 5 个点: index 0, 2, 4, 6, 9 (首末+3个中间)
        sample_idx = [0, 2, 4, 6, 9]
        xs = [xs_full[i] for i in sample_idx]
        t0 = time.time()
        print(f"\n=== {sw_name} ({sw_cfg['xlabel']}) ===")

        results = []
        for i, x in enumerate(xs):
            r = run_one_point(
                sw_name, x, mc_runs=1, episodes=0, eval_runs=0,
                base_seed=2024, verbose=False,
                mat_path="matlab.mat", ml_dir="ml-100k")
            results.append(r)
            print(f"  [{i+1}/5] x={x:.3f}  QoE={r['qoe_kkt']:8.1f}  "
                  f"hit={r['hit']:.3f}  wait={r['wait']:.2f}")

        with open(f"cache/demo_{sw_name}.pkl", "wb") as f:
            pickle.dump(dict(sweep=sw_name, xlabel=sw_cfg['xlabel'],
                              results=results), f)

        plot_sweep(sw_name, results, save_dir=SAVE_DIR)
        print(f"  ({time.time()-t0:.0f}s)")

    print(f"\nTotal: {time.time()-total_start:.0f}s")
    print(f"Figures in {SAVE_DIR}/")


if __name__ == "__main__":
    main()
