# -*- coding: utf-8 -*-
"""
initial_wait.py
===============
完整对齐 cal_initial_wait_times.m 的 5 case 逻辑:

    Case 1: IU 用户自缓存               → 0
    Case 2: 本地 SBS 缓存                → v*/p_sbs + D_bf/r(t=0)
    Case 3: 社区内某 IU 缓存 且距离 ≤ iu_coverage → v*/p_iu + D_bf/r(t=0)
    Case 4: 其他 SBS 缓存                → t_prop + v*/p_sbs + D_bf/r(t=0)
    Case 5: 云端                         → t_cloud + v*/p_sbs + D_bf/r(t=0)
"""
import numpy as np
import config as cfg


def compute_initial_wait_times(
    topo, iu_indices, iu_flags, cache_decision,
    community_users, requested_videos, download_rates,
    target_community=cfg.TARGET_COMMUNITY
):
    N_comu = len(community_users)
    m0 = target_community - 1
    total_users = cfg.TOTAL_USERS
    iu_per_comm = cfg.IU_PER_COMMUNITY
    init_wait = np.zeros(N_comu, dtype=np.float64)

    for i, user_idx in enumerate(community_users):
        req = requested_videos[i]
        r0 = max(download_rates[i, 0], 1e-3)

        # ---- Case 1: IU 自缓存 ----
        if iu_flags[user_idx] == 1 and cache_decision[user_idx, req] == 1:
            init_wait[i] = 0.0
            continue

        # ---- Case 2: 本地 SBS ----
        # cache_decision 后 H 行对应 SBS (索引 total_users + m0)
        if cache_decision[total_users + m0, req] == 1:
            init_wait[i] = cfg.V_CHI_STAR / cfg.P_SBS_COMP + cfg.D_BF / r0
            continue

        # ---- Case 3: 社区内某 IU 缓存 ----
        found_iu = False
        for j in range(iu_per_comm):
            iu_idx = int(iu_indices[m0, j])
            diff = (topo.now_user_positions[user_idx, :, 0]
                    - topo.now_user_positions[iu_idx,   :, 0])
            dist = np.linalg.norm(diff)
            if cache_decision[iu_idx, req] == 1 and dist <= cfg.IU_COVERAGE:
                init_wait[i] = cfg.V_CHI_STAR / cfg.P_IU_COMP + cfg.D_BF / r0
                found_iu = True
                break
        if found_iu:
            continue

        # ---- Case 4: 其他 SBS ----
        found_other = False
        for mm in range(cfg.H):
            if mm == m0:
                continue
            if cache_decision[total_users + mm, req] == 1:
                init_wait[i] = (cfg.T_PROPAGATION
                                + cfg.V_CHI_STAR / cfg.P_SBS_COMP
                                + cfg.D_BF / r0)
                found_other = True
                break
        if found_other:
            continue

        # ---- Case 5: 云端 ----
        init_wait[i] = (cfg.T_CLOUD
                        + cfg.V_CHI_STAR / cfg.P_SBS_COMP
                        + cfg.D_BF / r0)

    return init_wait
