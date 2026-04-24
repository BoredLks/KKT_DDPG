# -*- coding: utf-8 -*-
"""
download_rates.py
=================
完整复刻 MATLAB cal_download_rates_task_assignment.m 的活代码段 (L446-687):

    Step 1: IU 自身命中 → 本地服务,download_rate = R_max,task_assignment = 其在 iu_indices 中的位置
    Step 2: 枚举所有 UE→IU 候选对 (dist ≤ iu_coverage,IU 缓存了文件),计算带干扰的 D2D 速率
    Step 3: 按速率降序贪心分配 (每 IU 对外只服务 1 个用户,每 UE 只匹配 1 个 IU)
    Step 4: 剩余未匹配 UE → 退回 SBS,计算 SBS 带干扰速率;速率阈值 0.5 Mbps

输出:
    download_rates  (N_comu, T)
    task_assignment (N_comu, T)   0 = SBS,>0 = IU 局部编号 (1..iu_count_per_community)
"""
import numpy as np
import config as cfg
from channel import channel_gain, sinr_rate, r_max_sbs


def compute_download_rates_task_assignment(
    topo, iu_indices, iu_flags, cache_decision,
    community_users, requested_videos,
    target_community=cfg.TARGET_COMMUNITY,
    rng=None
):
    """
    参数说明(全部 0-based 索引):
        topo              : Topology
        iu_indices        : (H, iu_per_comm) 每行=每社区的 IU 全局 id (0-based)
        iu_flags          : (total_users,) 0/1
        cache_decision    : (total_users + H, F) 前 total_users 行是 user/IU,
                            后 H 行是 SBS
        community_users   : (N_comu,) 目标社区用户的全局 0-based 索引
        requested_videos  : (N_comu,) 每个用户请求的视频 id (0-based)
        target_community  : 1-based(与 MATLAB 一致),内部转 0-based 用
    """
    if rng is None:
        rng = np.random.RandomState(0)

    T = cfg.T_SMALL
    N_comu = len(community_users)
    iu_per_comm = cfg.IU_PER_COMMUNITY
    m0 = target_community - 1                       # 0-based 社区索引
    R_MAX = r_max_sbs()

    download_rates  = np.zeros((N_comu, T), dtype=np.float64)
    task_assignment = np.zeros((N_comu, T), dtype=np.int64)

    target_iu_row = iu_indices[m0]                  # 该社区 IU 全局 id 列表

    for t in range(T):
        iu_busy_other = np.zeros(iu_per_comm, dtype=bool)

        # ---- Step 1: IU 自身命中 ----
        for i, user_idx in enumerate(community_users):
            req = requested_videos[i]
            if iu_flags[user_idx] == 1 and cache_decision[user_idx, req] == 1:
                download_rates[i, t] = R_MAX
                # 在目标社区 IU 行内查找该 user_idx 的局部位置 (1-based)
                hit = np.where(target_iu_row == user_idx)[0]
                if len(hit) > 0:
                    task_assignment[i, t] = int(hit[0]) + 1

        # ---- Step 2: 枚举所有 UE→IU 候选对 ----
        cands = []                                   # (rate, i, j_local)
        for i, user_idx in enumerate(community_users):
            if download_rates[i, t] > 0:
                continue                             # 自服务已填
            req = requested_videos[i]
            for j in range(iu_per_comm):
                iu_idx = int(target_iu_row[j])
                if cache_decision[iu_idx, req] != 1:
                    continue
                diff = (topo.now_user_positions[user_idx, :, t]
                        - topo.now_user_positions[iu_idx,   :, t])
                dist = np.linalg.norm(diff)
                if dist > cfg.IU_COVERAGE:
                    continue

                # 信道增益
                g = channel_gain(dist, rng)

                # 同社区其他 IU 造成的干扰
                interference = 0.0
                for k in range(iu_per_comm):
                    other_iu = int(target_iu_row[k])
                    if other_iu == iu_idx or other_iu == user_idx:
                        continue
                    diff2 = (topo.now_user_positions[other_iu, :, t]
                             - topo.now_user_positions[user_idx, :, t])
                    d_int = np.linalg.norm(diff2)
                    if d_int <= cfg.IU_COVERAGE:
                        g_int = channel_gain(d_int, rng)
                        interference += cfg.P_IU * g_int

                rate = sinr_rate(cfg.P_IU * g, interference)
                if rate > 0.5:
                    cands.append((rate, i, j))

        # ---- Step 3: 按速率降序贪心 ----
        if cands:
            cands.sort(key=lambda x: -x[0])
            assigned_users = np.zeros(N_comu, dtype=bool)
            for rate, i, j in cands:
                if (not assigned_users[i]) and (not iu_busy_other[j]):
                    download_rates[i, t] = rate
                    task_assignment[i, t] = j + 1    # 1-based
                    assigned_users[i] = True
                    iu_busy_other[j] = True

        # ---- Step 4: 剩下的退回 SBS ----
        for i, user_idx in enumerate(community_users):
            if download_rates[i, t] > 0:
                continue
            sbs_pos = topo.sbs_positions[m0]
            up = topo.now_user_positions[user_idx, :, t]
            d_sbs = np.linalg.norm(up - sbs_pos)
            if d_sbs <= cfg.SBS_COVERAGE:
                g = channel_gain(d_sbs, rng)
                # 其他 SBS 干扰
                interference = 0.0
                for mm in range(cfg.H):
                    if mm == m0:
                        continue
                    d_int = np.linalg.norm(up - topo.sbs_positions[mm])
                    g_int = channel_gain(d_int, rng)
                    interference += cfg.P_SBS * g_int
                rate = sinr_rate(cfg.P_SBS * g, interference)
                if rate < 0.5:
                    rate = 0.5
                download_rates[i, t] = rate
            task_assignment[i, t] = 0

    return download_rates, task_assignment
