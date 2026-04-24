# -*- coding: utf-8 -*-
"""
preference.py
=============
多图融合缓存偏好,完整对齐 MATLAB:

- cal_cache_preference.m (公式 40):
    pSPAG[i, f] = (1/|N+(vi)|) · Σ_{vj ∈ N+(vi), vj 是用户}
                  q_vj,f · e_joint[vi, vj]
    只对"用户邻居"求和(MATLAB 的 if j ≤ total_users)

- cal_pF.m (公式 41):
    avg_sim_f    = mean(FAG_sim[f, neighbors])
    pF[i, f]     = q_vi,f · avg_sim_f                   (用户节点)
    pF[sbs_m, f] = mean(q_vi,f for i in community m) · avg_sim_f  (SBS节点)

- cal_fused_preference.m (公式 42):
    fused[i, f] = σ(θ · (β·pSPAG + (1-β)·pF))
"""
import numpy as np
import config as cfg


# ------------------------------------------------------------
def cal_cache_preference(user_file_req_prob, joint_edge_weights):
    """pSPAG[i, f],对齐 cal_cache_preference.m."""
    N = cfg.TOTAL_USERS
    H = cfg.H
    F = cfg.F_FILES
    pref = np.zeros((N + H, F), dtype=np.float64)

    for i in range(N + H):
        neighbors = np.where(joint_edge_weights[i] > 0)[0]
        # 只保留"用户"邻居(索引 < N)
        user_neighbors = neighbors[neighbors < N]
        if len(user_neighbors) == 0:
            continue
        div = len(neighbors)    # MATLAB: length(out_neighbors),包含 SBS
        for f in range(F):
            s = 0.0
            for j in user_neighbors:
                s += user_file_req_prob[j, f] * joint_edge_weights[i, j]
            pref[i, f] = s / div
    return pref


# ------------------------------------------------------------
def cal_p_fag(user_file_req_prob, FAG_sim):
    """pF[i, f],对齐 cal_pF.m."""
    N = cfg.TOTAL_USERS
    H = cfg.H
    F = cfg.F_FILES
    users_per_comm = cfg.USERS_PER_COMMUNITY

    # 每个文件的平均 FAG 相似度
    avg_sim_per_file = np.zeros(F, dtype=np.float64)
    for f in range(F):
        neighbors = np.where(FAG_sim[f] > 0)[0]
        if len(neighbors) > 0:
            avg_sim_per_file[f] = FAG_sim[f, neighbors].sum() / len(neighbors)

    p_fag = np.zeros((N + H, F), dtype=np.float64)
    for i in range(N):
        for f in range(F):
            p_fag[i, f] = user_file_req_prob[i, f] * avg_sim_per_file[f]

    # SBS 节点: 用其社区的平均请求概率
    for m in range(H):
        com_start = m * users_per_comm
        com_end   = (m + 1) * users_per_comm
        avg_req = user_file_req_prob[com_start:com_end].mean(axis=0)
        p_fag[N + m] = avg_req * avg_sim_per_file
    return p_fag


# ------------------------------------------------------------
def cal_fused_preference(cache_preference, p_fag,
                         theta_sigmoid=cfg.THETA_SIGMOID,
                         beta_fuse=cfg.BETA_FUSE):
    """Sigmoid 融合,对齐 cal_fused_preference.m (公式 42)."""
    combined = beta_fuse * cache_preference + (1 - beta_fuse) * p_fag
    # 数值稳定的 sigmoid
    x = theta_sigmoid * combined
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
