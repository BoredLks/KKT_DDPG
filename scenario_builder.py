# -*- coding: utf-8 -*-
"""
scenario_builder.py
===================
统一的场景构建器,支持 8 种参数扫描,与原 MATLAB 8 个脚本对齐:

    MATLAB 脚本          扫描变量     Python overrides 参数
    Com_Parameter        K            K_CH
    Everycom_Per_IU      Per_m        PER_M / IU_PER_COMMUNITY
    IU_CacheSize         D_iu         D_IU
    IU_Computing         p_iu         P_IU_COMP
    SBS_CacheSize        D_eg         D_EG
    SBS_Computing        p_sbs        P_SBS_COMP
    Users_Percom         user_per_com USERS_PER_COMMUNITY / TOTAL_USERS / IU_PER_COMMUNITY
    Zipf                 gamma_m      GAMMA_ZIPF

设计:
- 用上下文管理器 CfgOverride 暂时改 cfg 里的全局参数
- 构建完场景后参数自动恢复 (确保 smoke test 之间不互相污染)
"""
import os
import copy
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
from hit_rate           import compute_hit_rate


# ============================================================
#  cfg 覆盖上下文管理器
# ============================================================
class CfgOverride(object):
    """临时覆盖 cfg 里的全局参数,退出时恢复."""
    def __init__(self, **kwargs):
        self.overrides = kwargs
        self.saved = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            if hasattr(cfg, k):
                self.saved[k] = getattr(cfg, k)
                setattr(cfg, k, v)
        return self

    def __exit__(self, *args):
        for k, v in self.saved.items():
            setattr(cfg, k, v)


# ============================================================
#  核心 build_scenario
# ============================================================
def build_scenario(seed, overrides=None,
                    mat_path="matlab.mat", ml_dir="ml-100k",
                    use_random_iu=False):
    """
    参数:
        seed      : RandomState 种子
        overrides : dict,cfg 参数覆盖,例如 {"D_EG": 2e5}
        use_random_iu : Random-IU baseline

    返回一个 dict,字段与之前 run_sweep.build_scenario 一致。
    """
    if overrides is None:
        overrides = {}

    with CfgOverride(**overrides):
        rng = np.random.RandomState(seed)

        # --- topology ---
        topo = Topology(mat_path=mat_path, seed=seed)

        # 若 USERS_PER_COMMUNITY/TOTAL_USERS 有改动,位置数据需要截取前 N 个
        N_need = cfg.TOTAL_USERS
        if topo.total_users != N_need:
            topo = _truncate_topology(topo, N_need)

        # --- MovieLens ---
        R, _, num_movies_ml = load_udata(os.path.join(ml_dir, "u.data"))
        sim_matrix = cal_movielens(R)
        genre_matrix_ml, release_years_ml, num_genres = cal_load_item_data(
            os.path.join(ml_dir, "u.item"), num_movies_ml)

        sg_edge_weights = build_sg_edge_weights(topo, sim_matrix, rng)
        user_file_req_prob = cal_user_file_req_prob(
            R, sg_edge_weights, gamma_m=cfg.GAMMA_ZIPF, rng=rng)

        # --- IU 选择 ---
        if use_random_iu:
            iu_indices = np.zeros((cfg.H, cfg.IU_PER_COMMUNITY), dtype=np.int64)
            iu_flags = np.zeros(cfg.TOTAL_USERS, dtype=np.int64)
            for m in range(cfg.H):
                local = np.arange(m * cfg.USERS_PER_COMMUNITY,
                                  (m + 1) * cfg.USERS_PER_COMMUNITY)
                chosen = rng.choice(local, size=cfg.IU_PER_COMMUNITY,
                                     replace=False)
                chosen.sort()
                iu_indices[m] = chosen
                iu_flags[chosen] = 1
        else:
            iu_indices, iu_flags = select_iu_mois(topo, sg_edge_weights, rng)

        # --- 其余构建 ---
        pl_edge_weights = build_pl_edge_weights(topo, iu_indices, iu_flags, rng)
        joint_edge_weights = pl_edge_weights * sg_edge_weights
        cache_preference = cal_cache_preference(user_file_req_prob,
                                                  joint_edge_weights)

        file_ml_map = np.mod(np.arange(cfg.F_FILES), num_movies_ml)
        FAG_sim = cal_fag_sim(cfg.F_FILES, num_movies_ml, num_genres,
                               genre_matrix_ml, release_years_ml, file_ml_map)
        p_fag = cal_p_fag(user_file_req_prob, FAG_sim)
        fused_preference = cal_fused_preference(cache_preference, p_fag)

        cache_decision = cal_cache_decision_mois(
            fused_preference, iu_indices,
            D_eg=cfg.D_EG, D_iu=cfg.D_IU)

        community_users = topo.community_members(cfg.TARGET_COMMUNITY)
        requested_videos = cal_requested_videos_enhanced(
            user_file_req_prob, FAG_sim, sg_edge_weights, sim_matrix,
            topo.user_community, community_users, cfg.TARGET_COMMUNITY,
            rng=rng)

        download_rates, task_assignment = \
            compute_download_rates_task_assignment(
                topo, iu_indices, iu_flags, cache_decision,
                community_users, requested_videos,
                rng=np.random.RandomState(seed + 111))
        initial_wait_times = compute_initial_wait_times(
            topo, iu_indices, iu_flags, cache_decision,
            community_users, requested_videos, download_rates)

        hit_total, hit_iu, hit_sbs = compute_hit_rate(
            topo, iu_indices, iu_flags, cache_decision,
            community_users, requested_videos)

        # 返回快照(此时 cfg 还是 override 状态,数据里的形状等都一致)
        return dict(
            community_users=community_users,
            requested_videos=requested_videos,
            download_rates=download_rates,
            task_assignment=task_assignment,
            iu_flags=iu_flags,
            cache_decision=cache_decision,
            initial_wait_times=initial_wait_times,
            hit_rate=hit_total,
            iu_hit_rate=hit_iu,
            sbs_hit_rate=hit_sbs,
            mean_init_wait=float(initial_wait_times.mean()),
            # 保留此场景构建时的参数,供 KKT/DDPG 调用时继续锁定
            overrides=dict(overrides),
        )


# ============================================================
#  辅助: 用户数变化时的位置截取
# ============================================================
def _truncate_topology(topo, N_need):
    """
    当 USERS_PER_COMMUNITY 被改小时,从 per_user_positions / now_user_positions
    中按社区截取前 N_need/H 个用户。
    原始 matlab.mat 里 user_per_community 是 50,只能截小不能放大。
    """
    import copy as _copy
    new_topo = _copy.copy(topo)
    H = cfg.H
    upc_new = N_need // H
    upc_old = topo.total_users // H

    if upc_new > upc_old:
        raise ValueError(
            f"USERS_PER_COMMUNITY={upc_new} 超过原始 {upc_old},"
            "无法从 matlab.mat 截取。请考虑 Topology 合成模式。")

    keep_idx = []
    for m in range(H):
        start = m * upc_old
        keep_idx.extend(range(start, start + upc_new))
    keep_idx = np.array(keep_idx)

    new_topo.total_users = N_need
    new_topo.user_community = topo.user_community[keep_idx]
    new_topo.now_user_positions = topo.now_user_positions[keep_idx]
    new_topo.per_user_positions = topo.per_user_positions[keep_idx]
    new_topo.users_per_comm = upc_new
    return new_topo
