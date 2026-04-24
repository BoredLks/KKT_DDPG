# -*- coding: utf-8 -*-
"""
abr_kkt.py
==========
你 MATLAB cal_final_alltime_qoe.m 的 Python 版本,作为 DDPG 的对比 baseline。

严格对齐 MATLAB:
- Stage A (L47-162): 拉格朗日迭代
    * φ = ω4·Δt/(BF+ξ) + λ + μ                (公式 52)
    * z = (−b + √(b²+8ω1ω2)) / (4ω2)           (公式 63)
    * 截断到 [1e-6, download_rate]              (公式 62)
    * λ, μ_sbs, μ_iu 次梯度更新                 (公式 64)
- Stage B (L165-210): 离散投影,higher/lower 选 QoE 更高者,且截断到 R_cap
- Stage C (L215-335): SBS 与各 IU 容量约束,按 qoe_loss 降序逐步降级

is_self_cached 的用户(IU 自缓存)直接置 chi(end),不参与 λ/μ 迭代。
"""
import numpy as np
import config as cfg


# ------------------------------------------------------------
def _qoe_new(bf_cur, z, z_prev, wait, w1, w2, w3, w4, dt, xi):
    """公式 (29) 单用户 QoE."""
    z = max(z, 1e-6)
    return (w1 * np.log(z)
            - w2 * (z - z_prev) ** 2
            - w3 * wait
            - w4 * z * dt / (bf_cur + xi))


# ------------------------------------------------------------
def run_abr_kkt(
    community_users, requested_videos, download_rates, task_assignment,
    iu_flags, cache_decision, initial_wait_times,
    eta=cfg.ETA, eps_conv=cfg.EPSILON_CONV, max_iter=cfg.MAX_ITERATIONS,
    chi=cfg.CHI_MBPS, nf=cfg.N_F, D_bf=cfg.D_BF,
    p_sbs=cfg.P_SBS_COMP, p_iu=cfg.P_IU_COMP,
    dt=cfg.DELTA_T, xi=cfg.XI_BUF,
    w1=cfg.ALPHA_QOE, w2=cfg.BETA_QOE, w3=cfg.GAMMA_QOE, w4=cfg.DELTA_QOE,
    iu_count_per_community=cfg.IU_PER_COMMUNITY,
    rng=None,
):
    """返回 (final_alltime_qoe, r_decision, buffer_state)."""
    if rng is None:
        rng = np.random.RandomState(0)

    Nu = len(community_users)
    # ---- 初始化决策 / 状态(对应 cal_Initialize_optimization_variables.m) ----
    r_decision = np.full((Nu, nf), chi[0], dtype=np.float64)
    r_previous = np.zeros((Nu, nf), dtype=np.float64)
    lam     = rng.rand(Nu, nf)
    mu_sbs  = rng.rand(nf)
    mu_iu   = rng.rand(iu_count_per_community, nf)
    buffer_state = np.zeros((Nu, nf), dtype=np.float64)
    buffer_state[:, 0] = D_bf

    # self_cached 的用户全程置 chi[-1]
    is_self = np.zeros(Nu, dtype=bool)
    for i, uid in enumerate(community_users):
        if iu_flags[uid] == 1 and cache_decision[uid, requested_videos[i]] == 1:
            is_self[i] = True
            r_decision[i, :] = chi[-1]

    wait_cur = initial_wait_times.copy()        # 每次 t>0 会被清 0
    total_qoe = 0.0

    # ============================================================
    #  主循环:每 t 独立求解
    # ============================================================
    for t in range(nf):
        # ---- 时隙间状态更新 (L33-44) ----
        if t == 0:
            for i in range(Nu):
                r_previous[i, 0] = chi[rng.randint(len(chi))]
        else:
            r_previous[:, t] = r_decision[:, t - 1]
            wait_cur[:] = 0.0
            for i in range(Nu):
                buffer_state[i, t] = max(0.0, min(D_bf,
                    buffer_state[i, t - 1]
                    + (download_rates[i, t - 1] - r_decision[i, t - 1]) * dt))

        # ---- Stage A: 拉格朗日迭代 ----
        lagrange_hist = []
        for it in range(max_iter):
            # ---- 计算当前负载 ----
            sbs_load = 0.0
            iu_load = np.zeros(iu_count_per_community)
            for n in range(Nu):
                if task_assignment[n, t] == 0 and not is_self[n]:
                    sbs_load += r_decision[n, t]
                elif task_assignment[n, t] > 0 and not is_self[n]:
                    iu_load[task_assignment[n, t] - 1] += r_decision[n, t]

            # ---- 计算拉格朗日值 ----
            L_val = 0.0
            for i in range(Nu):
                z_cur = max(r_decision[i, t], 1e-6)
                qoe_i = _qoe_new(buffer_state[i, t], z_cur,
                                 r_previous[i, t], wait_cur[i],
                                 w1, w2, w3, w4, dt, xi)
                lam_pen = lam[i, t] * (r_decision[i, t] - download_rates[i, t])
                L_val += qoe_i - lam_pen
            L_val -= mu_sbs[t] * (sbs_load - p_sbs)
            for n in range(iu_count_per_community):
                L_val -= mu_iu[n, t] * (iu_load[n] - p_iu)

            lagrange_hist.append(L_val)
            if len(lagrange_hist) > 1 and abs(lagrange_hist[-1] - lagrange_hist[-2]) < eps_conv:
                break

            # ---- 步骤1: 公式 (52)(63) 更新 z ----
            for i in range(Nu):
                if is_self[i]:
                    r_decision[i, t] = chi[-1]
                    continue
                bf = buffer_state[i, t]
                # 选 μ 值
                if task_assignment[i, t] == 0:
                    mu_val = mu_sbs[t]
                else:
                    mu_val = mu_iu[task_assignment[i, t] - 1, t]
                phi = w4 * dt / (bf + xi) + lam[i, t] + mu_val
                b_coef = phi - 2 * w2 * r_previous[i, t]
                disc = b_coef ** 2 + 8 * w1 * w2
                r_new = (-b_coef + np.sqrt(disc)) / (4 * w2)
                r_new = min(r_new, download_rates[i, t])
                r_new = max(r_new, 1e-6)
                r_decision[i, t] = r_new

            # ---- 步骤2: 公式 (64) 更新对偶变量 ----
            for i in range(Nu):
                lam[i, t] = max(0.0,
                    lam[i, t] + eta * (r_decision[i, t] - download_rates[i, t]))

            sbs_load = 0.0
            iu_load = np.zeros(iu_count_per_community)
            for n in range(Nu):
                if task_assignment[n, t] == 0 and not is_self[n]:
                    sbs_load += r_decision[n, t]
                elif task_assignment[n, t] > 0 and not is_self[n]:
                    iu_load[task_assignment[n, t] - 1] += r_decision[n, t]
            mu_sbs[t] = max(0.0, mu_sbs[t] + eta * (sbs_load - p_sbs))
            for n in range(iu_count_per_community):
                mu_iu[n, t] = max(0.0, mu_iu[n, t] + eta * (iu_load[n] - p_iu))

        # ============================================================
        #  Stage B: 离散投影 (L165-210)
        # ============================================================
        r_final = np.zeros(Nu)
        for i in range(Nu):
            bf = buffer_state[i, t]
            R_cap = download_rates[i, t]
            z_cont = r_decision[i, t]

            if z_cont <= chi[0]:
                lower = higher = chi[0]
            elif z_cont >= chi[-1]:
                lower = higher = chi[-1]
            else:
                ups = np.where(chi >= z_cont)[0]
                if len(ups) == 0 or ups[0] == 0:
                    lower = higher = chi[0]
                else:
                    lower = chi[ups[0] - 1]
                    higher = chi[ups[0]]

            # FIX-1: 截断到 download_rate
            higher = min(higher, R_cap)
            lower  = min(lower,  R_cap)
            if abs(higher - lower) < 1e-9:
                r_final[i] = higher
            else:
                q_h = _qoe_new(bf, higher, r_previous[i, t], wait_cur[i],
                               w1, w2, w3, w4, dt, xi)
                q_l = _qoe_new(bf, lower,  r_previous[i, t], wait_cur[i],
                               w1, w2, w3, w4, dt, xi)
                r_final[i] = higher if q_h >= q_l else lower

        # ============================================================
        #  Stage C: 计算容量约束降级 (L215-335)
        # ============================================================

        # ---- C1: SBS 容量 ----
        sbs_users = [i for i in range(Nu) if task_assignment[i, t] == 0]
        sbs_users_nt = [i for i in sbs_users
                        if r_final[i] < chi[-1] and not is_self[i]]
        if sbs_users_nt:
            total_load = sum(r_final[i] for i in sbs_users_nt)
            if total_load > p_sbs:
                losses = []
                for i in sbs_users_nt:
                    idx_now = np.where(np.abs(chi - r_final[i]) < 1e-9)[0]
                    if len(idx_now) == 0 or idx_now[0] == 0:
                        continue
                    low_res = chi[idx_now[0] - 1]
                    q_now = _qoe_new(buffer_state[i, t], r_final[i],
                                     r_previous[i, t], wait_cur[i],
                                     w1, w2, w3, w4, dt, xi)
                    q_low = _qoe_new(buffer_state[i, t], low_res,
                                     r_previous[i, t], wait_cur[i],
                                     w1, w2, w3, w4, dt, xi)
                    losses.append((q_low - q_now, i, low_res))
                # 按 qoe_loss 降序(损失小的先降 → 差值最大的 loss 最负/最小 → 降序等价 MATLAB 的 'descend')
                losses.sort(key=lambda x: -x[0])
                for loss, i, low_res in losses:
                    r_final[i] = low_res
                    new_load = sum(r_final[j] for j in sbs_users_nt
                                   if r_final[j] < chi[-1])
                    if new_load <= p_sbs:
                        break

        # ---- C2: 每个 IU 容量 ----
        for j in range(iu_count_per_community):
            iu_users = [i for i in range(Nu) if task_assignment[i, t] == j + 1]
            iu_users_nt = [i for i in iu_users
                           if r_final[i] < chi[-1] and not is_self[i]]
            if not iu_users_nt:
                continue
            total_load = sum(r_final[i] for i in iu_users_nt)
            if total_load <= p_iu:
                continue
            losses = []
            for i in iu_users_nt:
                idx_now = np.where(np.abs(chi - r_final[i]) < 1e-9)[0]
                if len(idx_now) == 0 or idx_now[0] == 0:
                    continue
                low_res = chi[idx_now[0] - 1]
                q_now = _qoe_new(buffer_state[i, t], r_final[i],
                                 r_previous[i, t], wait_cur[i],
                                 w1, w2, w3, w4, dt, xi)
                q_low = _qoe_new(buffer_state[i, t], low_res,
                                 r_previous[i, t], wait_cur[i],
                                 w1, w2, w3, w4, dt, xi)
                losses.append((q_low - q_now, i, low_res))
            losses.sort(key=lambda x: -x[0])
            for loss, i, low_res in losses:
                r_final[i] = low_res
                new_load = sum(r_final[k] for k in iu_users_nt
                               if r_final[k] < chi[-1])
                if new_load <= p_iu:
                    break

        r_decision[:, t] = r_final

        # ---- t 时刻最终 QoE ----
        for i in range(Nu):
            total_qoe += _qoe_new(buffer_state[i, t], r_final[i],
                                  r_previous[i, t], wait_cur[i],
                                  w1, w2, w3, w4, dt, xi)

    return total_qoe, r_decision, buffer_state
