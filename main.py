# -*- coding: utf-8 -*-
"""
main.py
=======
完整对齐 algorithm_MOIS.m 的 20 步流程 (现已全部实现,无占位):

    Step 1.  位置数据                                 -> topology.py
    Step 2.  MovieLens 用户相似度 sim_matrix           -> movielens.cal_movielens
    Step 3.  电影属性(genre+year) 加载                 -> movielens.cal_load_item_data
    Step 4.  社交感知图 sg_edge_weights                -> graphs.build_sg_edge_weights
    Step 5.  用户文件请求概率 (Zipf+社交平滑)            -> user_req_prob.cal_user_file_req_prob
    Step 6.  MOIS 选择 IU                             -> mois.select_iu_mois
    Step 7.  物理连接图 pl_edge_weights                -> graphs.build_pl_edge_weights
    Step 8.  联合图 E = E^pg .* E^sg (公式 39)          -> numpy 逐元素乘
    Step 9.  pSPAG 缓存偏好 (公式 40)                  -> preference.cal_cache_preference
    Step 10. FAG 文件属性图 (公式 15-17)               -> movielens.cal_fag_sim
    Step 11. pF 文件偏好 (公式 41)                     -> preference.cal_p_fag
    Step 12. Sigmoid 融合 (公式 42)                    -> preference.cal_fused_preference
    Step 13. 缓存决策                                  -> cache_decision.cal_cache_decision_mois
    Step 14. 目标社区用户                              -> topology.community_members
    Step 15. 增强版请求生成                            -> requested_videos.cal_requested_videos_enhanced
    Step 16. 下载速率 + 任务分配                       -> download_rates
    Step 17. 初始等待时间                              -> initial_wait
    Step 18. (初始化优化变量已嵌在 abr_kkt / abr_ddpg 内部)
    Step 19. QoE 优化 (KKT 或 DDPG 二选一)              -> abr_kkt / abr_ddpg
    Step 20. 缓存命中率                                -> hit_rate

Usage:
    python main.py --mode both --mat matlab.mat --episodes 500
    python main.py --mode kkt
    python main.py --mode ddpg --episodes 300
"""
import os
import argparse
import time
import numpy as np

import config as cfg
from topology           import Topology
from movielens          import (load_udata, cal_movielens,
                                 cal_load_item_data, cal_fag_sim)
from graphs             import build_sg_edge_weights, build_pl_edge_weights
from user_req_prob      import cal_user_file_req_prob
from mois               import select_iu_mois
from preference         import (cal_cache_preference, cal_p_fag,
                                 cal_fused_preference)
from cache_decision     import cal_cache_decision_mois
from requested_videos   import cal_requested_videos_enhanced
from download_rates     import compute_download_rates_task_assignment
from initial_wait       import compute_initial_wait_times
from abr_kkt            import run_abr_kkt
from abr_ddpg           import run_abr_ddpg
from hit_rate           import compute_hit_rate


def run_pipeline(mat_path=None, ml_dir="ml-100k",
                 seed=cfg.SEED_DEFAULT, verbose=True):
    """
    跑一遍 algorithm_MOIS.m 的 Step 1-17 + Step 20,
    返回公共输入给 KKT 和 DDPG 选择其一跑 Step 19.
    """
    if verbose:
        print(f"[Pipeline] seed={seed}")
    t0 = time.time()
    rng = np.random.RandomState(seed)

    # -------- Step 1: topology --------
    topo = Topology(mat_path=mat_path, seed=seed)
    if verbose:
        print(f"  Step 1  topology  users={topo.total_users}  T={cfg.T_SMALL}")

    # -------- Step 2: MovieLens 相似度 --------
    u_data_path = os.path.join(ml_dir, "u.data")
    u_item_path = os.path.join(ml_dir, "u.item")
    if not os.path.exists(u_data_path):
        raise FileNotFoundError(
            f"MovieLens data not found at {u_data_path}. "
            f"Place ml-100k/ under working dir or pass --ml-dir.")
    R, num_users_ml, num_movies_ml = load_udata(u_data_path)
    sim_matrix = cal_movielens(R)
    if verbose:
        print(f"  Step 2  MovieLens R shape={R.shape}, "
              f"sim_matrix shape={sim_matrix.shape}")

    # -------- Step 3: 电影属性 --------
    genre_matrix_ml, release_years_ml, num_genres = cal_load_item_data(
        u_item_path, num_movies_ml)
    if verbose:
        print(f"  Step 3  loaded {num_genres} genres for {num_movies_ml} movies")

    # -------- Step 4: 社交感知图 --------
    sg_edge_weights = build_sg_edge_weights(topo, sim_matrix, rng)
    if verbose:
        print(f"  Step 4  SG edges (mean) = {sg_edge_weights.mean():.4f}")

    # -------- Step 5: 用户文件请求概率 --------
    user_file_req_prob = cal_user_file_req_prob(
        R, sg_edge_weights, rng=rng)
    if verbose:
        print(f"  Step 5  user_file_req_prob shape={user_file_req_prob.shape}, "
              f"row_sum≈{user_file_req_prob.sum(axis=1).mean():.3f}")

    # -------- Step 6: MOIS 选 IU --------
    iu_indices, iu_flags = select_iu_mois(topo, sg_edge_weights, rng)
    if verbose:
        print(f"  Step 6  iu_indices shape={iu_indices.shape}, "
              f"iu_flags sum={iu_flags.sum()}")

    # -------- Step 7: 物理连接图 --------
    pl_edge_weights = build_pl_edge_weights(topo, iu_indices, iu_flags, rng)
    if verbose:
        print(f"  Step 7  PL edges (mean) = {pl_edge_weights.mean():.4f}")

    # -------- Step 8: 联合图 (公式 39 逐元素乘) --------
    joint_edge_weights = pl_edge_weights * sg_edge_weights
    if verbose:
        print(f"  Step 8  joint edges (mean) = {joint_edge_weights.mean():.4f}")

    # -------- Step 9: pSPAG --------
    cache_preference = cal_cache_preference(user_file_req_prob,
                                              joint_edge_weights)
    if verbose:
        print(f"  Step 9  pSPAG mean = {cache_preference.mean():.6f}")

    # -------- Step 10: FAG --------
    file_ml_map = np.mod(np.arange(cfg.F_FILES), num_movies_ml)
    FAG_sim = cal_fag_sim(cfg.F_FILES, num_movies_ml, num_genres,
                           genre_matrix_ml, release_years_ml, file_ml_map)
    if verbose:
        print(f"  Step 10 FAG_sim nonzero frac = "
              f"{(FAG_sim > 0).mean():.3f}")

    # -------- Step 11: pF --------
    p_fag = cal_p_fag(user_file_req_prob, FAG_sim)
    if verbose:
        print(f"  Step 11 pF mean = {p_fag.mean():.6f}")

    # -------- Step 12: 融合 --------
    fused_preference = cal_fused_preference(cache_preference, p_fag)
    if verbose:
        print(f"  Step 12 fused_pref mean = {fused_preference.mean():.4f}")

    # -------- Step 13: 缓存决策 --------
    cache_decision = cal_cache_decision_mois(fused_preference, iu_indices)
    if verbose:
        sbs_cached = cache_decision[cfg.TOTAL_USERS:].sum()
        iu_cached = cache_decision[:cfg.TOTAL_USERS].sum()
        print(f"  Step 13 cache_decision: SBS entries={sbs_cached}, "
              f"IU entries={iu_cached}")

    # -------- Step 14: 目标社区用户 --------
    community_users = topo.community_members(cfg.TARGET_COMMUNITY)
    if verbose:
        print(f"  Step 14 target community={cfg.TARGET_COMMUNITY}, "
              f"|community_users|={len(community_users)}")

    # -------- Step 15: 增强请求生成 --------
    requested_videos = cal_requested_videos_enhanced(
        user_file_req_prob, FAG_sim, sg_edge_weights, sim_matrix,
        topo.user_community, community_users, cfg.TARGET_COMMUNITY, rng=rng)
    if verbose:
        uniq, cnts = np.unique(requested_videos, return_counts=True)
        print(f"  Step 15 requested_videos unique={len(uniq)}, "
              f"top-1 count={cnts.max()}")

    # -------- Step 16: 下载速率 + 任务分配 --------
    if verbose:
        print("  Step 16 computing download rates & task assignment ...")
    rng_rates = np.random.RandomState(seed + 111)
    download_rates, task_assignment = compute_download_rates_task_assignment(
        topo, iu_indices, iu_flags, cache_decision,
        community_users, requested_videos, rng=rng_rates)
    if verbose:
        print(f"  Step 16 mean download_rate = {download_rates.mean():.2f} Mbps")

    # -------- Step 17: 初始等待时间 --------
    initial_wait_times = compute_initial_wait_times(
        topo, iu_indices, iu_flags, cache_decision,
        community_users, requested_videos, download_rates)
    if verbose:
        print(f"  Step 17 mean initial wait = "
              f"{initial_wait_times.mean():.3f} s")

    # -------- Step 20: 命中率 --------
    hit_total, hit_iu, hit_sbs = compute_hit_rate(
        topo, iu_indices, iu_flags, cache_decision,
        community_users, requested_videos)
    if verbose:
        print(f"  Step 20 hit rate = {hit_total:.3f} "
              f"(iu={hit_iu:.3f}, sbs={hit_sbs:.3f})")
        print(f"  Pipeline built in {time.time()-t0:.1f}s")

    return dict(
        topo=topo,
        iu_indices=iu_indices, iu_flags=iu_flags,
        cache_decision=cache_decision,
        community_users=community_users,
        requested_videos=requested_videos,
        download_rates=download_rates,
        task_assignment=task_assignment,
        initial_wait_times=initial_wait_times,
        hit_rate=hit_total,
        iu_hit_rate=hit_iu, sbs_hit_rate=hit_sbs,
        # 附带中间变量,方便调试
        sg_edge_weights=sg_edge_weights,
        pl_edge_weights=pl_edge_weights,
        joint_edge_weights=joint_edge_weights,
        fused_preference=fused_preference,
        user_file_req_prob=user_file_req_prob,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both",
                        choices=["kkt", "ddpg", "both"])
    parser.add_argument("--episodes", type=int, default=cfg.NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=cfg.SEED_DEFAULT)
    parser.add_argument("--mat", type=str, default="matlab.mat",
                        help="MATLAB .mat file path (optional)")
    parser.add_argument("--ml-dir", type=str, default="ml-100k",
                        help="MovieLens 100k directory")
    args = parser.parse_args()

    mat_path = args.mat if os.path.exists(args.mat) else None
    if mat_path is None:
        print(f"[Note] {args.mat} not found, synthesizing topology.")

    ctx = run_pipeline(mat_path=mat_path, ml_dir=args.ml_dir, seed=args.seed)

    qoe_kkt = qoe_ddpg = None

    # ---------- KKT ----------
    if args.mode in ("kkt", "both"):
        print("\n" + "=" * 60)
        print("[KKT] running cal_final_alltime_qoe (Algorithm 2) ...")
        t0 = time.time()
        rng_kkt = np.random.RandomState(args.seed + 11)
        qoe_kkt, r_kkt, bf_kkt = run_abr_kkt(
            community_users=ctx['community_users'],
            requested_videos=ctx['requested_videos'],
            download_rates=ctx['download_rates'],
            task_assignment=ctx['task_assignment'],
            iu_flags=ctx['iu_flags'],
            cache_decision=ctx['cache_decision'],
            initial_wait_times=ctx['initial_wait_times'],
            rng=rng_kkt)
        print(f"[KKT]  Total QoE = {qoe_kkt:.3f}  ({time.time()-t0:.1f}s)")
        print(f"[KKT]  Mean last-10-slot bitrate = {r_kkt[:, -10:].mean():.2f} Mbps")
        print(f"[KKT]  Mean last-10-slot buffer  = {bf_kkt[:, -10:].mean():.2f} Mbit")

    # ---------- DDPG ----------
    if args.mode in ("ddpg", "both"):
        print("\n" + "=" * 60)
        print(f"[DDPG] training for {args.episodes} episodes ...")
        t0 = time.time()
        rng_ddpg = np.random.RandomState(args.seed + 22)
        agent, rewards, qoe_ddpg, r_ddpg, bf_ddpg = run_abr_ddpg(
            community_users=ctx['community_users'],
            requested_videos=ctx['requested_videos'],
            download_rates=ctx['download_rates'],
            task_assignment=ctx['task_assignment'],
            iu_flags=ctx['iu_flags'],
            cache_decision=ctx['cache_decision'],
            initial_wait_times=ctx['initial_wait_times'],
            num_episodes=args.episodes,
            rng=rng_ddpg, verbose=True)
        print(f"[DDPG] Eval Total QoE = {qoe_ddpg:.3f}  ({time.time()-t0:.1f}s)")
        print(f"[DDPG] Mean last-10-slot bitrate = {r_ddpg[:, -10:].mean():.2f} Mbps")
        print(f"[DDPG] First→Last ep reward: {rewards[0]:.1f} → {rewards[-1]:.1f}")

    # ---------- Summary ----------
    if args.mode == "both":
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Hit rate    : {ctx['hit_rate']:.3f}  "
              f"(iu={ctx['iu_hit_rate']:.3f}, sbs={ctx['sbs_hit_rate']:.3f})")
        print(f"  KKT  QoE    : {qoe_kkt:.3f}")
        print(f"  DDPG QoE    : {qoe_ddpg:.3f}")
        gap = (qoe_ddpg - qoe_kkt) / max(abs(qoe_kkt), 1e-6) * 100
        print(f"  Gap (DDPG-KKT)/|KKT| = {gap:+.1f}%")


if __name__ == "__main__":
    main()
