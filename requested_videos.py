# -*- coding: utf-8 -*-
"""
requested_videos.py
===================
完整对齐 cal_requested_videos_enhanced.m 的增强请求生成:

两个增强因子:
  Social_boost[i, f] = Σ_{top_k friends} w_friend · q_friend,f
                       其中 w_friend = 0.5*sg_weight + 0.5*ps_weight
                       行最大值归一化
  FAG_boost[i, f]    = mean_{f' 是 f 的 FAG 邻居}
                        q_user,f' · FAG_sim[f, f']
                       行最大值归一化

融合(乘法增强):
  enhanced[i, f] = q_base[u,f]
                   · (1 + λ_s · Social_boost[i,f])
                   · (1 + λ_f · FAG_boost[i,f])
  再按行归一化为概率分布

最后按 cumulative CDF 采样每个用户的请求视频.
"""
import numpy as np
import config as cfg


def cal_requested_videos_enhanced(
    user_file_req_prob, FAG_sim, sg_edge_weights, sim_matrix,
    user_community, community_users, target_community,
    lambda_social_req=0.8, lambda_fag_req=0.5, top_k_friends=5,
    rng=None,
):
    """
    参数:
        user_file_req_prob  (N, F)
        FAG_sim             (F, F)
        sg_edge_weights     (N+H, N+H)
        sim_matrix          (num_users_ml, num_users_ml) MovieLens 相似度
        user_community      (N,) 1-based 社区编号
        community_users     (N_comu,) 目标社区用户全局 0-based id
        target_community    1-based
    返回:
        requested_videos    (N_comu,) 0-based 视频 id
    """
    if rng is None:
        rng = np.random.RandomState(0)

    F = cfg.F_FILES
    N = cfg.TOTAL_USERS
    num_users_ml = sim_matrix.shape[0]
    N_comu = len(community_users)

    user_ml_map = np.mod(np.arange(N), num_users_ml)

    # ---- 为每个目标社区用户生成朋友集合(社区内随机 50%) ----
    user_friends = {}
    com_members = np.where(user_community == target_community)[0]
    num_friends = max(1, min(len(com_members),
                              int(round(len(com_members) * 0.5))))
    for u in community_users:
        user_friends[int(u)] = rng.choice(com_members,
                                           size=num_friends,
                                           replace=False)

    # ---- 因子 1: Social_boost ----
    Social_boost = np.zeros((N_comu, F), dtype=np.float64)
    for i, uid in enumerate(community_users):
        friends = user_friends[int(uid)]
        friends_in_com = friends[user_community[friends] == target_community]
        if len(friends_in_com) == 0:
            continue

        sg_w = sg_edge_weights[uid, friends_in_com]
        mi = user_ml_map[uid]
        ps_w = np.zeros(len(friends_in_com))
        for jj, fj in enumerate(friends_in_com):
            mj = user_ml_map[fj]
            ps_w[jj] = sim_matrix[mi, mj]
        combined_w = 0.5 * sg_w + 0.5 * ps_w

        k = min(top_k_friends, len(friends_in_com))
        top_idx = np.argsort(-combined_w)[:k]
        top_friends = friends_in_com[top_idx]
        top_weights = combined_w[top_idx]

        for jj in range(k):
            Social_boost[i] += top_weights[jj] * user_file_req_prob[top_friends[jj]]

    # 行最大值归一化
    for i in range(N_comu):
        mx = Social_boost[i].max()
        if mx > 0:
            Social_boost[i] = Social_boost[i] / mx

    # ---- 因子 2: FAG_boost ----
    FAG_boost = np.zeros((N_comu, F), dtype=np.float64)
    for i, uid in enumerate(community_users):
        for f in range(F):
            neighbors = np.where(FAG_sim[f] > 0)[0]
            if len(neighbors) > 0:
                boost = np.sum(user_file_req_prob[uid, neighbors]
                               * FAG_sim[f, neighbors])
                FAG_boost[i, f] = boost / len(neighbors)

    for i in range(N_comu):
        mx = FAG_boost[i].max()
        if mx > 0:
            FAG_boost[i] = FAG_boost[i] / mx

    # ---- 乘法融合 ----
    enhanced = np.zeros((N_comu, F), dtype=np.float64)
    for i, uid in enumerate(community_users):
        q_base = user_file_req_prob[uid]
        s_factor = 1.0 + lambda_social_req * Social_boost[i]
        f_factor = 1.0 + lambda_fag_req * FAG_boost[i]
        enh = q_base * s_factor * f_factor
        total = enh.sum()
        if total > 0:
            enhanced[i] = enh / total
        else:
            enhanced[i] = q_base

    # ---- 按 CDF 采样 ----
    requested = np.zeros(N_comu, dtype=np.int64)
    for i in range(N_comu):
        cdf = np.cumsum(enhanced[i])
        r = rng.rand()
        idx = np.searchsorted(cdf, r)
        requested[i] = min(idx, F - 1)
    return requested
