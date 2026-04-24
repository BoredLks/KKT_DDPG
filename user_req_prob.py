# -*- coding: utf-8 -*-
"""
user_req_prob.py
================
完整对齐 cal_user_file_req_prob.m 的两阶段流程:

Stage 1: 基于 MovieLens 评分的个性化 Zipf
  - 每系统用户映射到一个 MovieLens 用户 (mod 取模)
  - 每系统文件映射到一个 MovieLens 电影
  - 文件兴趣分数:已评分则用评分,未评分用"电影平均评分+小随机噪声"
  - 按分数降序排名,Zipf 分布赋概率

Stage 2: 社交图平滑(固定点迭代)
  - W_social 行归一化
  - P^{t+1} = (1-social_blend)·P_zipf + social_blend·W·P^t
  - 每轮重新归一化(非负,行和=1)
  - 迭代直到 Frobenius 范数差 < tol 或达到 social_iter_max
"""
import numpy as np
import config as cfg


def cal_user_file_req_prob(R, sg_edge_weights,
                            total_users=cfg.TOTAL_USERS,
                            F=cfg.F_FILES,
                            gamma_m=cfg.GAMMA_ZIPF,
                            social_iter_max=100,
                            social_blend=0.35,
                            social_tol=1e-8,
                            rng=None):
    """
    参数:
        R            : (num_users_ml, num_movies_ml) MovieLens 评分矩阵
        sg_edge_weights : (total_users+H, total_users+H) 社交图
        rng          : 随机数生成器(用于未评分电影的小扰动)
    返回:
        user_file_req_prob : (total_users, F)  每行和为 1
    """
    if rng is None:
        rng = np.random.RandomState(0)

    num_users_ml, num_movies_ml = R.shape
    user_ml_map = np.mod(np.arange(total_users), num_users_ml)   # 0-based
    file_ml_map = np.mod(np.arange(F), num_movies_ml)

    # ---- 预计算每部电影的全局平均评分(只考虑非零) ----
    movie_mean = np.zeros(num_movies_ml)
    for mid in range(num_movies_ml):
        col = R[:, mid]
        nz = col[col > 0]
        if nz.size > 0:
            movie_mean[mid] = nz.mean()
        else:
            movie_mean[mid] = 2.5       # 论文默认

    # 全局平均(兜底)
    all_nz = R[R > 0]
    global_mean = all_nz.mean() if all_nz.size > 0 else 2.5

    # ---- Stage 1: Zipf 概率 ----
    P_zipf = np.zeros((total_users, F), dtype=np.float64)
    zipf_denom = np.sum(np.arange(1, F + 1, dtype=np.float64) ** (-gamma_m))

    for u in range(total_users):
        ml_u = user_ml_map[u]
        user_ratings = R[ml_u]
        file_scores = np.zeros(F)
        for k in range(F):
            ml_k = file_ml_map[k]
            r = user_ratings[ml_k]
            if r > 0:
                file_scores[k] = r
            else:
                mm = movie_mean[ml_k] if movie_mean[ml_k] > 0 else global_mean
                file_scores[k] = mm + 0.1 * rng.randn()     # 小噪声

        # 按分数降序,赋 Zipf 概率
        # MATLAB: phi_k_ui(rank) = file_id; prob(k) = rank(k)^(-γ) / Σ
        sorted_idx = np.argsort(-file_scores)    # 从高到低的文件 id
        rank_of_file = np.zeros(F, dtype=np.int64)
        for rank_pos, fid in enumerate(sorted_idx):
            rank_of_file[fid] = rank_pos + 1     # 1-based rank
        P_zipf[u] = (rank_of_file.astype(np.float64) ** (-gamma_m)) / zipf_denom

    # ---- Stage 2: 社交图平滑 ----
    W_social = sg_edge_weights[:total_users, :total_users].copy()
    W_social[W_social < 0] = 0.0
    row_sum = W_social.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    W_norm = W_social / row_sum

    P_social = P_zipf.copy()
    for it in range(social_iter_max):
        P_next = (1 - social_blend) * P_zipf + social_blend * (W_norm @ P_social)
        P_next[P_next < 0] = 0.0
        rs = P_next.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        P_next = P_next / rs
        if np.linalg.norm(P_next - P_social, 'fro') < social_tol:
            P_social = P_next
            break
        P_social = P_next

    return P_social
