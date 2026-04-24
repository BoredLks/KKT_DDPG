# -*- coding: utf-8 -*-
"""
mois.py
=======
完整对齐 cal_Select_iu_MOIS.m 的 MOIS 多对一匹配博弈 + 空间分散后处理。

流程 (246 行 MATLAB 压缩为 ~180 行 Python):
  6.1 预备 D2D 评分 (所有社区内用户两两采样平均速率,R_max_pre 归一化)
  6.2 逐社区:
       a) 公式 (36) 重要性 W_ij = μ·e^pg + ν·e^sg + υ·c (c=min电量)
       b) 租金 S_rent ~ rent_base·(0.5+sum_w/max_sw)·(0.9+0.2·rand)
          归一化到 [0,1]
       c) S_CP[cp,ii] = ζ·Σ W_ij(j in N(ii)) - S_rent[cp,ii]
          PR_CP 排序
       d) S_UE[ii,cp] = S_rent[cp,ii] - S_cost
       e) MOIS 匹配主循环: CP 按 PR 提 proposal,UE 在多 proposal 时选
          S_UE 最高的 CP,被拒 CP 继续对下一排名 UE 提议,直到 CP 名额满
       f) 空间分散贪心:从 MOIS 候选 + 全部用户中,按综合分贪心选,
          每轮选能新覆盖最多未覆盖用户的候选
"""
import numpy as np
import config as cfg
from channel import channel_gain


def _prelim_d2d_score(topo, rng):
    """预备物理评分 (社区内两两 D2D 速率平均,记录 R_max_pre)."""
    N = cfg.TOTAL_USERS
    H = cfg.H
    users_per_comm = cfg.USERS_PER_COMMUNITY
    T = cfg.T_SMALL
    score = np.zeros((N, N), dtype=np.float64)
    R_max_pre = 0.0

    for m in range(H):
        com_start = m * users_per_comm
        com_end   = (m + 1) * users_per_comm
        for i in range(com_start, com_end):
            for j in range(i + 1, com_end):
                total = 0.0
                valid = 0
                for t in range(T):
                    d = np.linalg.norm(
                        topo.per_user_positions[i, :, t]
                        - topo.per_user_positions[j, :, t])
                    if d > cfg.IU_COVERAGE or d <= 0:
                        continue
                    g = channel_gain(d, rng)
                    snr = cfg.P_IU * g / cfg.N0
                    snr = max(min(snr, 1e12), 0.0)
                    R_inst = cfg.B_MHZ * np.log2(1.0 + snr)
                    if R_inst > R_max_pre:
                        R_max_pre = R_inst
                    total += R_inst
                    valid += 1
                if valid > 0:
                    avg = total / valid
                    score[i, j] = avg
                    score[j, i] = avg

    if R_max_pre > 0:
        for i in range(N):
            score[i, i] = R_max_pre
        return score / R_max_pre, score       # 归一化 + 原始
    return np.zeros_like(score), score


# ------------------------------------------------------------
def select_iu_mois(topo, sg_edge_weights, rng,
                   C_num=3,
                   mu_mois=0.4, nu_mois=0.4, upsilon_mois=0.2,
                   zeta_coeff=1.0, S_cost_val=0.5,
                   rent_base=None):
    """完整 MOIS + 空间分散,返回 (iu_indices[H, iu_per_comm], iu_flags[N])."""
    N = cfg.TOTAL_USERS
    H = cfg.H
    users_per_comm = cfg.USERS_PER_COMMUNITY
    iu_per_comm = cfg.IU_PER_COMMUNITY

    if rent_base is None:
        rent_base = np.array([1.5, 2.0, 1.8])

    # Q_cp: CP 名额分配 (iu_per_comm 尽量均分 C_num 个 CP)
    Q_base = iu_per_comm // C_num
    Q_rem  = iu_per_comm % C_num
    Q_cp   = np.full(C_num, Q_base, dtype=np.int64)
    for cp in range(Q_rem):
        Q_cp[cp] += 1

    # 电量均匀 [0.5, 1]
    battery = 0.5 + 0.5 * rng.rand(N)

    # 6.1 D2D 预备评分
    prelim_d2d_norm, _ = _prelim_d2d_score(topo, rng)

    # 平均位置,用于空间分散
    avg_pos = topo.per_user_positions.mean(axis=2)

    iu_indices = np.zeros((H, iu_per_comm), dtype=np.int64)
    iu_flags = np.zeros(N, dtype=np.int64)

    for m in range(H):
        com_start = m * users_per_comm
        com_end   = (m + 1) * users_per_comm
        com_users = np.arange(com_start, com_end)      # 全局 0-based
        N_com = users_per_comm

        # ------ 公式 (36) 重要性 W_importance ------
        W_imp = np.zeros((N_com, N_com))
        for ii in range(N_com):
            gi = com_users[ii]
            for jj in range(N_com):
                if ii == jj:
                    continue
                gj = com_users[jj]
                e_pg = prelim_d2d_norm[gi, gj]
                e_sg = sg_edge_weights[gi, gj]
                c_ij = min(battery[gi], battery[gj])
                W_imp[ii, jj] = (mu_mois * e_pg
                                 + nu_mois * e_sg
                                 + upsilon_mois * c_ij)
        sum_w = W_imp.sum(axis=1)
        max_sw = sum_w.max() if sum_w.max() > 0 else 1.0

        # D2D 邻居(原始速率 > 0)
        # 对齐 MATLAB: d2d_check = prelim_d2d_score(com,com)
        d2d_check = np.zeros((N_com, N_com))
        for ii in range(N_com):
            for jj in range(N_com):
                if ii != jj:
                    d2d_check[ii, jj] = prelim_d2d_norm[com_users[ii],
                                                        com_users[jj]]
        N_neighbors = [np.where(d2d_check[ii] > 0)[0] for ii in range(N_com)]

        # ------ CP 租金 S_rent ------
        S_rent = np.zeros((C_num, N_com))
        for cp in range(C_num):
            for ii in range(N_com):
                S_rent[cp, ii] = (rent_base[cp]
                                   * (0.5 + sum_w[ii] / max_sw)
                                   * (0.9 + 0.2 * rng.rand()))
        mn, mx = S_rent.min(), S_rent.max()
        if mx > mn:
            S_rent = (S_rent - mn) / (mx - mn)
        else:
            S_rent = np.ones_like(S_rent)

        # ------ S_CP 与 PR_CP ------
        S_CP = np.zeros((C_num, N_com))
        for cp in range(C_num):
            for ii in range(N_com):
                nb = N_neighbors[ii]
                sum_w_nb = W_imp[ii, nb].sum() if nb.size > 0 else 0.0
                S_CP[cp, ii] = zeta_coeff * sum_w_nb - S_rent[cp, ii]
        PR_CP = np.zeros((C_num, N_com), dtype=np.int64)
        for cp in range(C_num):
            PR_CP[cp] = np.argsort(-S_CP[cp])            # 降序

        # ------ S_UE ------
        S_UE = np.zeros((N_com, C_num))
        for ii in range(N_com):
            for cp in range(C_num):
                S_UE[ii, cp] = S_rent[cp, ii] - S_cost_val

        # ------ MOIS 匹配主循环 ------
        Y = np.zeros((C_num, N_com), dtype=np.int64)
        cp_count = np.zeros(C_num, dtype=np.int64)
        ue_owner = np.full(N_com, -1, dtype=np.int64)     # -1 = 未匹配
        cp_reject = [set() for _ in range(C_num)]
        cp_next_rank = np.zeros(C_num, dtype=np.int64)

        max_iter = N_com * C_num + 50
        for it in range(max_iter):
            if (cp_count >= Q_cp).all():
                break
            proposals = [[] for _ in range(N_com)]
            any_p = False
            # 每个未满名额的 CP 发一个 proposal
            for cp in range(C_num):
                if cp_count[cp] >= Q_cp[cp]:
                    continue
                while cp_next_rank[cp] < N_com:
                    ue_l = int(PR_CP[cp, cp_next_rank[cp]])
                    cp_next_rank[cp] += 1
                    if ue_l in cp_reject[cp]:
                        continue
                    proposals[ue_l].append(cp)
                    any_p = True
                    break
            if not any_p:
                break

            # 每个 UE 处理收到的 proposal
            for ue_l in range(N_com):
                inc = proposals[ue_l]
                if not inc:
                    continue
                cur = ue_owner[ue_l]
                cands = list(inc)
                if cur >= 0:
                    cands.append(cur)
                # 选 S_UE 最高的 CP
                bcp = max(cands, key=lambda c: S_UE[ue_l, c])
                # 若之前有 owner 且被替换,释放 + 拒绝
                if cur >= 0 and bcp != cur:
                    Y[cur, ue_l] = 0
                    cp_count[cur] -= 1
                    cp_reject[cur].add(ue_l)
                if cur < 0 or bcp != cur:
                    ue_owner[ue_l] = bcp
                    Y[bcp, ue_l] = 1
                    cp_count[bcp] += 1
                # 其他提 proposal 但未中选的 CP 也被拒绝
                for c in inc:
                    if c != bcp:
                        cp_reject[c].add(ue_l)

        mois_selected = np.where(Y.sum(axis=0) > 0)[0]

        # ------ 空间分散贪心 ------
        cover_count = np.zeros(N_com, dtype=np.int64)
        for ii in range(N_com):
            gi = com_users[ii]
            cnt = 0
            for jj in range(N_com):
                if ii != jj:
                    gj = com_users[jj]
                    if np.linalg.norm(avg_pos[gi] - avg_pos[gj]) <= cfg.IU_COVERAGE:
                        cnt += 1
            cover_count[ii] = cnt

        importance_score = sum_w.copy()
        mois_bonus = np.zeros(N_com)
        if importance_score.max() > 0:
            mois_bonus[mois_selected] = importance_score.max() * 0.3
        max_cover = cover_count.max() if cover_count.max() > 0 else 1
        max_bonus = mois_bonus.max() if mois_bonus.max() > 0 else 1
        combined_score = (importance_score / max_sw
                          + mois_bonus / max_bonus
                          + cover_count / max_cover)

        selected_local = []
        covered = np.zeros(N_com, dtype=bool)
        remaining = list(range(N_com))
        for _ in range(iu_per_comm):
            if not remaining:
                break
            best_score = -np.inf
            best_idx_in_remaining = 0
            for idx, ii in enumerate(remaining):
                gi = com_users[ii]
                new_covered = 0
                for jj in range(N_com):
                    if not covered[jj]:
                        gj = com_users[jj]
                        if np.linalg.norm(avg_pos[gi] - avg_pos[gj]) <= cfg.IU_COVERAGE:
                            new_covered += 1
                sc = new_covered * (1 + combined_score[ii])
                if sc > best_score:
                    best_score = sc
                    best_idx_in_remaining = idx
            best_local = remaining[best_idx_in_remaining]
            selected_local.append(best_local)
            # 更新覆盖
            gi = com_users[best_local]
            for jj in range(N_com):
                gj = com_users[jj]
                if np.linalg.norm(avg_pos[gi] - avg_pos[gj]) <= cfg.IU_COVERAGE:
                    covered[jj] = True
            remaining.pop(best_idx_in_remaining)

        selected_global = com_users[np.array(selected_local, dtype=np.int64)]
        selected_global.sort()
        iu_indices[m] = selected_global
        iu_flags[selected_global] = 1

    return iu_indices, iu_flags
