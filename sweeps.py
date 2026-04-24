# -*- coding: utf-8 -*-
"""
sweeps.py
=========
完整对应 MATLAB 8 个扫描脚本的 Python 版本:

    MATLAB 脚本          扫描变量  number=1..10 公式      X 轴标签
    Com_Parameter        K         K = 0.2 + 0.1*n       Communication Parameter K
    Everycom_Per_IU      Per_m     Per_m = 0.08*n        Percent of IU (%)
    IU_CacheSize         D_iu      D_iu = 2*n files      Cache size of IU (Files)
    IU_Computing         p_iu      p_iu = 1.5*n Mbps     IU Computing Power (Mbit/s)
    SBS_CacheSize        D_eg      D_eg = 5*n files      Cache size of SBS (Files)
    SBS_Computing        p_sbs     p_sbs = 20*n Mbps     SBS Computing Power (Mbit/s)
    Users_Percom         upc       upc = 5*n             Number of users per community
    Zipf                 gamma_m   gamma_m = 0.1*n       Zipf gamma

每个扫描:
  - 10 个扫描点
  - 每点: KKT 跑 `mc_runs`(默认 500)次取平均;DDPG 训 `episodes` 轮后评估 `eval_runs` 次取平均
  - 3 张图: QoE / Hit rate / Waiting time (对齐 MATLAB main.fig / Hit_rate.fig / Waiting_time.fig)

Usage:
    # 单个扫描
    python sweeps.py --sweep Zipf --mc-runs 500 --episodes 500 --eval-runs 3

    # smoke test (所有 8 个扫描, 少 MC)
    python sweeps.py --sweep ALL --mc-runs 3 --episodes 30 --eval-runs 1

    # 列出所有可用扫描
    python sweeps.py --list
"""
import os
import argparse
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from scenario_builder import build_scenario
from abr_runners      import run_kkt, run_ddpg


# ============================================================
#  扫描配置表 (完整对齐 MATLAB)
# ============================================================
SWEEP_CONFIGS = {
    "Com_Parameter": dict(
        xlabel="Communication Parameter K",
        x_values=[0.2 + 0.1 * n for n in range(1, 11)],   # 0.3..1.2
        override_key="K_CH",
    ),
    "Everycom_Per_IU": dict(
        xlabel="Percent of IU (%)",
        x_values=[0.08 * n * 100 for n in range(1, 11)],  # 8..80  (%)
        override_key="PER_M",   # 实际存 0.08*n (0..1)
        x_to_override=lambda xv: xv / 100.0,              # xv% -> 0.xx
    ),
    "IU_CacheSize": dict(
        xlabel="Cache size of IU (Files)",
        x_values=[2 * n for n in range(1, 11)],           # 2..20  files
        override_key="D_IU",
        x_to_override=lambda xv: xv * cfg.N_F * cfg.V_CHI_STAR,   # files -> Mbit
    ),
    "IU_Computing": dict(
        xlabel="IU Computing Power (Mbit/s)",
        x_values=[1.5 * n for n in range(1, 11)],         # 1.5..15
        override_key="P_IU_COMP",
    ),
    "SBS_CacheSize": dict(
        xlabel="Cache size of SBS (Files)",
        x_values=[5 * n for n in range(1, 11)],           # 5..50 files
        override_key="D_EG",
        x_to_override=lambda xv: xv * cfg.N_F * cfg.V_CHI_STAR,
    ),
    "SBS_Computing": dict(
        xlabel="SBS Computing Power (Mbit/s)",
        x_values=[20 * n for n in range(1, 11)],          # 20..200
        override_key="P_SBS_COMP",
    ),
    "Users_Percom": dict(
        xlabel="Number of users per community",
        x_values=[5 * n for n in range(1, 11)],           # 5..50
        override_key="USERS_PER_COMMUNITY",
    ),
    "Zipf": dict(
        xlabel="Zipf gamma",
        x_values=[0.1 * n for n in range(1, 11)],         # 0.1..1.0
        override_key="GAMMA_ZIPF",
    ),
}


# ============================================================
#  构建 overrides dict (含派生参数,比如 USERS_PER_COMMUNITY 改时 TOTAL_USERS/IU_PER_COMMUNITY 也要改)
# ============================================================
def build_overrides_for_point(sweep_name, x_value):
    cfg_sw = SWEEP_CONFIGS[sweep_name]
    key = cfg_sw["override_key"]
    val = cfg_sw.get("x_to_override", lambda v: v)(x_value)
    overrides = {key: val}

    # 派生参数处理
    if key == "PER_M":
        # IU 数量 = round(upc * Per_m)
        upc = cfg.USERS_PER_COMMUNITY
        overrides["IU_PER_COMMUNITY"] = int(round(upc * val))
    elif key == "USERS_PER_COMMUNITY":
        upc_new = int(val)
        overrides["USERS_PER_COMMUNITY"] = upc_new
        overrides["TOTAL_USERS"] = cfg.H * upc_new
        overrides["IU_PER_COMMUNITY"] = int(round(upc_new * cfg.PER_M))
    return overrides


# ============================================================
#  单次扫描: 返回 (qoe_mean, hit_mean, wait_mean)
# ============================================================
def run_one_point(sweep_name, x_value, mc_runs, episodes, eval_runs,
                   base_seed=2024, verbose=False,
                   mat_path="matlab.mat", ml_dir="ml-100k"):
    overrides = build_overrides_for_point(sweep_name, x_value)

    # --- KKT: 跑 mc_runs 次,每次不同 seed,取平均 ---
    qoe_kkt_runs = []
    hit_runs = []
    wait_runs = []
    t0 = time.time()
    for r in range(mc_runs):
        sd = base_seed + r * 37
        scen = build_scenario(seed=sd, overrides=overrides,
                               mat_path=mat_path, ml_dir=ml_dir)
        q = run_kkt(scen, seed=sd)
        qoe_kkt_runs.append(q)
        hit_runs.append(scen['hit_rate'])
        wait_runs.append(scen['mean_init_wait'])
    qoe_kkt = float(np.mean(qoe_kkt_runs))
    hit     = float(np.mean(hit_runs))
    wait    = float(np.mean(wait_runs))
    t_kkt = time.time() - t0

    # --- DDPG: 训 episodes 轮,跑 eval_runs 次不同 seed,每次返回单次训练后的 eval qoe ---
    t0 = time.time()
    qoe_ddpg_runs = []
    if episodes > 0:
        for r in range(eval_runs):
            sd = base_seed + r * 101 + 50000
            # 用 base scenario (不同 seed) 重新构建 scenario
            scen = build_scenario(seed=sd, overrides=overrides,
                                   mat_path=mat_path, ml_dir=ml_dir)
            q = run_ddpg(scen, episodes=episodes, eval_runs=1,
                          seed=sd, verbose=False)
            qoe_ddpg_runs.append(q)
        qoe_ddpg = float(np.mean(qoe_ddpg_runs))
    else:
        qoe_ddpg = float("nan")
    t_ddpg = time.time() - t0

    if verbose:
        print(f"    x={x_value:7.3f}  KKT={qoe_kkt:8.2f} "
              f"(mc={mc_runs}, {t_kkt:.1f}s)  "
              f"DDPG={qoe_ddpg:8.2f} "
              f"(ep={episodes},ev={eval_runs}, {t_ddpg:.1f}s)  "
              f"hit={hit:.3f}  wait={wait:.2f}")

    return dict(
        x=x_value,
        qoe_kkt=qoe_kkt, qoe_ddpg=qoe_ddpg,
        hit=hit, wait=wait,
        qoe_kkt_std=float(np.std(qoe_kkt_runs)),
        qoe_ddpg_std=float(np.std(qoe_ddpg_runs)) if qoe_ddpg_runs else 0.0,
    )


# ============================================================
#  执行整个 sweep
# ============================================================
def run_sweep(sweep_name, mc_runs=500, episodes=500, eval_runs=3,
              base_seed=2024, mat_path="matlab.mat", ml_dir="ml-100k",
              save_dir="Figure", verbose=True):
    cfg_sw = SWEEP_CONFIGS[sweep_name]
    xs = cfg_sw["x_values"]
    print(f"\n{'='*60}\n  Sweep: {sweep_name}  |  {cfg_sw['xlabel']}\n"
          f"  points={len(xs)}  mc_runs={mc_runs}  episodes={episodes}  "
          f"eval_runs={eval_runs}\n{'='*60}")

    results = []
    t_start = time.time()
    for pi, x in enumerate(xs):
        if verbose:
            print(f"  [{pi+1}/{len(xs)}] x={x}")
        r = run_one_point(sweep_name, x, mc_runs, episodes, eval_runs,
                           base_seed=base_seed + pi * 1000,
                           verbose=verbose,
                           mat_path=mat_path, ml_dir=ml_dir)
        results.append(r)

    print(f"\n  Sweep total: {time.time()-t_start:.1f}s")

    # 保存结果
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/{sweep_name}.pkl", "wb") as f:
        pickle.dump(dict(sweep=sweep_name, xlabel=cfg_sw['xlabel'],
                          results=results), f)

    # 绘图
    plot_sweep(sweep_name, results, save_dir=save_dir)

    return results


# ============================================================
#  绘图 (3 张图 per sweep)
# ============================================================
def plot_sweep(sweep_name, results, save_dir="Figure"):
    os.makedirs(os.path.join(save_dir, sweep_name), exist_ok=True)
    xs = [r["x"] for r in results]
    qoe_kkt  = [r["qoe_kkt"] for r in results]
    qoe_ddpg = [r["qoe_ddpg"] for r in results]
    hits     = [r["hit"] for r in results]
    waits    = [r["wait"] for r in results]
    xlabel = SWEEP_CONFIGS[sweep_name]["xlabel"]

    # --- 图 1: QoE ---
    plt.figure(figsize=(8, 6))
    plt.plot(xs, qoe_kkt, "-o", color="#d62728", lw=2.5, ms=8,
             label="Proposed (KKT)")
    if not np.all(np.isnan(qoe_ddpg)):
        plt.plot(xs, qoe_ddpg, "--s", color="#1f77b4", lw=2.5, ms=8,
                 label="Proposed (DDPG)")
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("QoE", fontsize=16)
    plt.xticks(xs, fontsize=11, rotation=30 if len(str(xs[0])) > 4 else 0)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=12, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, sweep_name, "main.png"), dpi=150)
    plt.close()

    # --- 图 2: Hit rate ---
    plt.figure(figsize=(8, 6))
    plt.plot(xs, hits, "-o", color="#d62728", lw=2.5, ms=8,
             label="Proposed")
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Hit rate", fontsize=16)
    plt.xticks(xs, fontsize=11, rotation=30 if len(str(xs[0])) > 4 else 0)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=12, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, sweep_name, "Hit_rate.png"), dpi=150)
    plt.close()

    # --- 图 3: Waiting time ---
    plt.figure(figsize=(8, 6))
    plt.plot(xs, waits, "-o", color="#d62728", lw=2.5, ms=8,
             label="Proposed")
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Waiting time (s)", fontsize=16)
    plt.xticks(xs, fontsize=11, rotation=30 if len(str(xs[0])) > 4 else 0)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=12, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, sweep_name, "Waiting_time.png"),
                 dpi=150)
    plt.close()


# ============================================================
#  CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=str, default="Zipf",
                    help="sweep name or 'ALL'")
    p.add_argument("--mc-runs", type=int, default=500,
                    help="KKT Monte-Carlo runs per point (MATLAB default 2000)")
    p.add_argument("--episodes", type=int, default=500,
                    help="DDPG training episodes per point")
    p.add_argument("--eval-runs", type=int, default=3,
                    help="DDPG eval runs per point")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--mat", type=str, default="matlab.mat")
    p.add_argument("--ml-dir", type=str, default="ml-100k")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        print("Available sweeps:")
        for name, info in SWEEP_CONFIGS.items():
            print(f"  {name:18s}  {info['xlabel']}  x={info['x_values']}")
        return

    mat_path = args.mat if os.path.exists(args.mat) else None
    if mat_path is None:
        print(f"[Note] {args.mat} not found, using synthesized topology.")

    sweeps = list(SWEEP_CONFIGS.keys()) if args.sweep == "ALL" else [args.sweep]
    for sw in sweeps:
        if sw not in SWEEP_CONFIGS:
            print(f"[Error] unknown sweep: {sw}")
            continue
        run_sweep(sw, mc_runs=args.mc_runs, episodes=args.episodes,
                   eval_runs=args.eval_runs, base_seed=args.seed,
                   mat_path=mat_path, ml_dir=args.ml_dir)
    print("\nAll done.")


if __name__ == "__main__":
    main()
