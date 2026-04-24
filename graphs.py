# -*- coding: utf-8 -*-
"""
graphs.py
=========
构建 SG(社交感知图)与 PL(物理连接图),严格对齐 MATLAB:
- cal_sg_edge_weights.m: α·IN + β·PS + γ·IM
- cal_pl_edge_weights.m (活代码 L380-561): 按社区对 D2D 速率采样求平均
"""
import numpy as np
import config as cfg
from channel import channel_gain


# ------------------------------------------------------------
def build_sg_edge_weights(topo, sim_matrix, rng):
    """
    对齐 cal_sg_edge_weights.m 的全部逻辑:
    - 每个用户在社区内随机选 50% 作为朋友集合 J_vi
    - 社交亲密度 IN  = |J_vi ∩ J_vj| / |J_vi ∪ J_vj|  (Jaccard, 公式 7)
    - 偏好相似度 PS  = sim_matrix[i,j]                 (公式 10)
    - 用户重要性 IM  = rand()                          (公式 13 的 MATLAB 简化)
    - 边权 = α·IN + β·PS + γ·IM                        (公式 14)
    - 自环边 = 1; 不同社区 = 0
    - SBS-用户: 用户在 SBS 覆盖内则边权 = 1,否则 = 0

    返回 (total_users + H, total_users + H) 矩阵
    """
    N = cfg.TOTAL_USERS
    H = cfg.H
    sz = N + H
    W = np.zeros((sz, sz), dtype=np.float64)

    # 用户平均位置
    user_avg_pos = topo.per_user_positions.mean(axis=2)      # (N, 2)

    # ---- 每用户的朋友集合 (社区内随机 50%) ----
    user_friends = [None] * N
    for u in range(N):
        com = topo.user_community[u]
        same_com = np.where(topo.user_community == com)[0]
        num_friends = max(1, min(len(same_com),
                                  int(round(len(same_com) * 0.5))))
        user_friends[u] = set(rng.choice(same_com, size=num_friends,
                                          replace=False).tolist())

    # ---- 用户对 ----
    for i in range(N):
        W[i, i] = 1.0                        # 自环
        com_i = topo.user_community[i]
        for j in range(i + 1, N):
            com_j = topo.user_community[j]
            if com_i != com_j:
                continue
            # 亲密度 IN
            ji = user_friends[i]
            jj = user_friends[j]
            inter = len(ji & jj)
            union = len(ji | jj)
            IN = inter / union if union > 0 else 0.0
            # 偏好相似度 PS (确保索引在 sim_matrix 范围内)
            if (i < sim_matrix.shape[0] and
                    j < sim_matrix.shape[0]):
                PS = float(sim_matrix[i, j])
            else:
                PS = 0.0
            # 重要性 IM
            IM = rng.rand()
            w = cfg.ALPHA_SG * IN + cfg.BETA_SG * PS + cfg.GAMMA_SG * IM
            W[i, j] = w
            W[j, i] = w

        # ---- 用户-SBS ----
        for m in range(H):
            sbs_idx = N + m
            d = np.linalg.norm(user_avg_pos[i] - topo.sbs_positions[m])
            if d <= cfg.SBS_COVERAGE:
                W[i, sbs_idx] = 1.0
                W[sbs_idx, i] = 1.0
    return W


# ------------------------------------------------------------
def build_pl_edge_weights(topo, iu_indices, iu_flags, rng):
    """
    对齐 cal_pl_edge_weights.m 活代码段 L380-561。
    返回 (N+H, N+H) 矩阵,值 ∈ [0,1].

    特征:
    - IU 自环 = 1,普通用户自环 = 0
    - 社区内用户对 (至少一方是 IU) 的 D2D 速率按 T_SMALL 个时隙采样求平均,
      然后 min(1, avg/R_max_iu) 归一化
    - 用户-本社区 SBS 同理用 R_max_sbs 归一化
    - SBS 自环 = 1,SBS 间边 = 0
    """
    N = cfg.TOTAL_USERS
    H = cfg.H
    users_per_comm = cfg.USERS_PER_COMMUNITY
    iu_per_comm = cfg.IU_PER_COMMUNITY
    T = cfg.T_SMALL

    W = np.zeros((N + H, N + H), dtype=np.float64)
    R_max_sbs = 0.0
    R_max_iu  = 0.0

    # 采样一整个 (i, j, t) 的 D2D 速率以及 SBS 速率,记录最大值以归一化
    # 为与 MATLAB 顺序完全一致,分两步:先算所有原始 rate,再归一化
    d2d_avg = {}        # (i, j) -> avg_rate (仅限同社区且至少一方 IU)
    sbs_avg = {}        # i -> avg_rate

    for m in range(H):
        com_start = m * users_per_comm
        com_end   = (m + 1) * users_per_comm
        target_iu_row = iu_indices[m]

        for i in range(com_start, com_end):
            # ---- D2D: (i, j) for j>i ----
            for j in range(i + 1, com_end):
                if not (iu_flags[i] == 1 or iu_flags[j] == 1):
                    continue
                total_rate = 0.0
                valid = 0
                for t in range(T):
                    d = np.linalg.norm(
                        topo.per_user_positions[i, :, t]
                        - topo.per_user_positions[j, :, t])
                    if d > cfg.IU_COVERAGE or d <= 0:
                        continue
                    g = channel_gain(d, rng)
                    # 干扰: 来自同社区其他 IU
                    interference = 0.0
                    for k in range(iu_per_comm):
                        oth = int(target_iu_row[k])
                        if oth == i or oth == j:
                            continue
                        d_int = np.linalg.norm(
                            topo.per_user_positions[oth, :, t]
                            - topo.per_user_positions[j,   :, t])
                        if d_int <= cfg.IU_COVERAGE:
                            g_int = channel_gain(d_int, rng)
                            interference += cfg.P_IU * g_int
                    snr = cfg.P_IU * g / (cfg.N0 + interference)
                    snr = max(min(snr, 1e12), 0.0)
                    R_inst = cfg.B_MHZ * np.log2(1.0 + snr)
                    if R_inst > R_max_iu:
                        R_max_iu = R_inst
                    total_rate += R_inst
                    valid += 1
                if valid > 0:
                    d2d_avg[(i, j)] = total_rate / valid

            # ---- 用户 i 到本社区 SBS ----
            total_rate = 0.0
            valid = 0
            for t in range(T):
                d = np.linalg.norm(
                    topo.per_user_positions[i, :, t]
                    - topo.sbs_positions[m])
                if d > cfg.SBS_COVERAGE or d <= 0:
                    continue
                g = channel_gain(d, rng)
                interference = 0.0
                for n in range(H):
                    if n == m:
                        continue
                    d_int = np.linalg.norm(
                        topo.per_user_positions[i, :, t]
                        - topo.sbs_positions[n])
                    g_int = channel_gain(d_int, rng)
                    interference += cfg.P_SBS * g_int
                snr = cfg.P_SBS * g / (cfg.N0 + interference)
                snr = max(min(snr, 1e12), 0.0)
                R_inst = cfg.B_MHZ * np.log2(1.0 + snr)
                if R_inst > R_max_sbs:
                    R_max_sbs = R_inst
                total_rate += R_inst
                valid += 1
            if valid > 0:
                sbs_avg[i] = total_rate / valid

    # ---- 写入归一化后的权重 ----
    for m in range(H):
        sbs_idx = N + m
        com_start = m * users_per_comm
        com_end   = (m + 1) * users_per_comm
        for i in range(com_start, com_end):
            if iu_flags[i] == 1:
                W[i, i] = 1.0
            else:
                W[i, i] = 0.0
            for j in range(i + 1, com_end):
                if (i, j) in d2d_avg:
                    if R_max_iu > 0:
                        w = min(1.0, d2d_avg[(i, j)] / R_max_iu)
                    else:
                        w = 0.0
                    W[i, j] = w
                    W[j, i] = w
            if i in sbs_avg:
                if R_max_sbs > 0:
                    w = min(1.0, sbs_avg[i] / R_max_sbs)
                else:
                    w = 0.0
                W[i, sbs_idx] = w
                W[sbs_idx, i] = w
        W[sbs_idx, sbs_idx] = 1.0
        for n in range(m + 1, H):
            W[sbs_idx, N + n] = 0.0
            W[N + n, sbs_idx] = 0.0

    return W
