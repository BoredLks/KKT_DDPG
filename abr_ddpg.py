# -*- coding: utf-8 -*-
"""
abr_ddpg.py
===========
用 DDPG 替换 cal_final_alltime_qoe.m 里的 Stage A (拉格朗日迭代) + Stage B (离散投影)。
Stage C (容量降级) 保留为同样的后处理,以保证约束被严格执行。

--- 环境定义 (与 KKT 版共用) ---
  State  : 每用户 [BF_i/D_bf, r_i/R_max, z_prev_i/χ*, task_assignment_onehot(SBS/IU)]
           维度 = Nu * (3 + 2) = Nu * 5  (Nu = 50)
  Action : 每用户 ∈ [-1,1],解码为 chi 中一个级别,维度 = Nu
  Reward : Σ QoE(29) - 约束违反惩罚

--- is_self 用户永远选 chi[-1],不进入 DDPG 输入/输出 ---

整体流程:
  for each episode:
      reset buffer_state & z_prev
      for t in range(nf):
          state = 构造 state(t)
          action = agent.select_action(state)
          z_idx = 解码 action 为 chi 索引
          r_cont = chi[z_idx]
          r_final = Stage B 投影(截断到 download_rate)
          r_final = Stage C 容量降级
          R(t) = Σ QoE_new(i) - PENALTY*(超限量)
          更新 buffer_state(t+1), z_prev(t+1)
          store (s,a,r,s',done)
          agent.update()

DDPG 输出 QoE(可与 KKT 直接比较)
"""
import numpy as np
import config as cfg
from ddpg_agent import DDPGAgent
from abr_kkt import _qoe_new


STATE_PER_USER = 5    # [bf_norm, r_norm, z_prev_norm, is_sbs, is_iu]


def _build_state(Nu, buffer_state_t, download_rates_t, z_prev_t,
                 task_assignment_t, D_bf, R_max, chi_star):
    feats = np.zeros(Nu * STATE_PER_USER, dtype=np.float32)
    for i in range(Nu):
        feats[i*5 + 0] = buffer_state_t[i] / D_bf
        feats[i*5 + 1] = min(download_rates_t[i] / max(R_max, 1e-3), 1.0)
        feats[i*5 + 2] = z_prev_t[i] / chi_star
        feats[i*5 + 3] = 1.0 if task_assignment_t[i] == 0 else 0.0
        feats[i*5 + 4] = 0.0 if task_assignment_t[i] == 0 else 1.0
    return feats


def _decode_action(action, chi, L):
    """连续 [-1,1] → chi 级别索引."""
    idx = np.round((action + 1.0) / 2.0 * (L - 1)).astype(np.int64)
    return np.clip(idx, 0, L - 1)


def _stage_bc_postprocess(
    r_cont, download_rates_t, task_assignment_t,
    is_self, z_prev_t, buffer_state_t, wait_t,
    chi, p_sbs, p_iu, iu_count, w1, w2, w3, w4, dt, xi, Nu
):
    """复用 KKT 版的 Stage B + Stage C 作为后处理."""
    # ---- Stage B: 截断到 chi 中且 ≤ download_rate ----
    r_final = np.zeros(Nu)
    for i in range(Nu):
        if is_self[i]:
            r_final[i] = chi[-1]
            continue
        z = r_cont[i]
        R_cap = download_rates_t[i]
        # 选 chi 中 ≤ R_cap 的最大级别作为上限
        feasible = chi[chi <= R_cap]
        if len(feasible) == 0:
            r_final[i] = chi[0]
        else:
            cap_val = feasible[-1]
            r_final[i] = min(z, cap_val)

    # ---- Stage C1: SBS 容量 ----
    sbs_nt = [i for i in range(Nu)
              if task_assignment_t[i] == 0 and not is_self[i]
              and r_final[i] < chi[-1]]
    if sbs_nt and sum(r_final[i] for i in sbs_nt) > p_sbs:
        losses = []
        for i in sbs_nt:
            idx_now = int(np.argmin(np.abs(chi - r_final[i])))
            if idx_now == 0:
                continue
            low = chi[idx_now - 1]
            q_now = _qoe_new(buffer_state_t[i], r_final[i], z_prev_t[i],
                             wait_t[i], w1, w2, w3, w4, dt, xi)
            q_low = _qoe_new(buffer_state_t[i], low, z_prev_t[i],
                             wait_t[i], w1, w2, w3, w4, dt, xi)
            losses.append((q_low - q_now, i, low))
        losses.sort(key=lambda x: -x[0])
        for _, i, low in losses:
            r_final[i] = low
            if sum(r_final[j] for j in sbs_nt if r_final[j] < chi[-1]) <= p_sbs:
                break

    # ---- Stage C2: 每个 IU 容量 ----
    for j in range(iu_count):
        iu_nt = [i for i in range(Nu)
                 if task_assignment_t[i] == j + 1 and not is_self[i]
                 and r_final[i] < chi[-1]]
        if not iu_nt or sum(r_final[i] for i in iu_nt) <= p_iu:
            continue
        losses = []
        for i in iu_nt:
            idx_now = int(np.argmin(np.abs(chi - r_final[i])))
            if idx_now == 0:
                continue
            low = chi[idx_now - 1]
            q_now = _qoe_new(buffer_state_t[i], r_final[i], z_prev_t[i],
                             wait_t[i], w1, w2, w3, w4, dt, xi)
            q_low = _qoe_new(buffer_state_t[i], low, z_prev_t[i],
                             wait_t[i], w1, w2, w3, w4, dt, xi)
            losses.append((q_low - q_now, i, low))
        losses.sort(key=lambda x: -x[0])
        for _, i, low in losses:
            r_final[i] = low
            if sum(r_final[k] for k in iu_nt if r_final[k] < chi[-1]) <= p_iu:
                break

    return r_final


def run_abr_ddpg(
    community_users, requested_videos, download_rates, task_assignment,
    iu_flags, cache_decision, initial_wait_times,
    agent=None,
    num_episodes=cfg.NUM_EPISODES, warmup=cfg.WARMUP_EPISODE,
    chi=cfg.CHI_MBPS, nf=cfg.N_F, D_bf=cfg.D_BF,
    p_sbs=cfg.P_SBS_COMP, p_iu=cfg.P_IU_COMP,
    dt=cfg.DELTA_T, xi=cfg.XI_BUF,
    w1=cfg.ALPHA_QOE, w2=cfg.BETA_QOE, w3=cfg.GAMMA_QOE, w4=cfg.DELTA_QOE,
    iu_count=cfg.IU_PER_COMMUNITY,
    penalty_coef=cfg.PENALTY_COEF,
    rng=None, verbose=False,
):
    """
    训练并评估 DDPG-ABR。
    Returns:
        agent           : 训练好的 DDPG agent
        episode_rewards : list[float] 每 ep 总 reward
        eval_qoe        : 最后一次 evaluation (关闭探索噪声) 的总 QoE
        r_decision_eval : (Nu, nf) 评估时的决策矩阵
        buffer_eval     : (Nu, nf) 评估时的 buffer 演化
    """
    if rng is None:
        rng = np.random.RandomState(0)

    Nu = len(community_users)
    L = len(chi)
    R_max = float(np.max(download_rates)) if download_rates.size > 0 else 1.0
    chi_star = float(chi[-1])

    # is_self
    is_self = np.zeros(Nu, dtype=bool)
    for i, uid in enumerate(community_users):
        if iu_flags[uid] == 1 and cache_decision[uid, requested_videos[i]] == 1:
            is_self[i] = True

    state_dim  = Nu * STATE_PER_USER
    action_dim = Nu

    if agent is None:
        agent = DDPGAgent(state_dim, action_dim)

    episode_rewards = []

    def run_one_episode(train_mode=True):
        buffer_state = np.zeros((Nu, nf))
        buffer_state[:, 0] = D_bf
        r_decision = np.full((Nu, nf), chi[0])
        r_previous = np.zeros((Nu, nf))
        for i in range(Nu):
            r_previous[i, 0] = chi[rng.randint(L)]
        wait_cur = initial_wait_times.copy()
        R_total = 0.0

        for t in range(nf):
            if t > 0:
                r_previous[:, t] = r_decision[:, t - 1]
                wait_cur[:] = 0.0
                for i in range(Nu):
                    buffer_state[i, t] = max(0.0, min(D_bf,
                        buffer_state[i, t - 1]
                        + (download_rates[i, t - 1]
                           - r_decision[i, t - 1]) * dt))

            # 构造 state
            s = _build_state(Nu, buffer_state[:, t], download_rates[:, t],
                             r_previous[:, t], task_assignment[:, t],
                             D_bf, R_max, chi_star)
            a = agent.select_action(s, explore=train_mode)
            z_idx = _decode_action(a, chi, L)
            r_cont = chi[z_idx]

            # is_self 强制最高
            for i in range(Nu):
                if is_self[i]:
                    r_cont[i] = chi[-1]

            # Stage B + C 后处理
            r_final = _stage_bc_postprocess(
                r_cont, download_rates[:, t], task_assignment[:, t],
                is_self, r_previous[:, t], buffer_state[:, t], wait_cur,
                chi, p_sbs, p_iu, iu_count, w1, w2, w3, w4, dt, xi, Nu
            )
            r_decision[:, t] = r_final

            # ---- 计算 reward ----
            qoe_sum = 0.0
            for i in range(Nu):
                qoe_sum += _qoe_new(buffer_state[i, t], r_final[i],
                                    r_previous[i, t], wait_cur[i],
                                    w1, w2, w3, w4, dt, xi)

            # 约束违反惩罚(DDPG 看到实际 r_cont 是否超限,引导它学得收敛更快)
            penalty = 0.0
            sbs_nt = [i for i in range(Nu)
                      if task_assignment[i, t] == 0 and not is_self[i]]
            over = sum(r_cont[i] for i in sbs_nt) - p_sbs
            if over > 0:
                penalty += penalty_coef * over
            for j in range(iu_count):
                iu_nt = [i for i in range(Nu)
                         if task_assignment[i, t] == j + 1 and not is_self[i]]
                over_j = sum(r_cont[i] for i in iu_nt) - p_iu
                if over_j > 0:
                    penalty += penalty_coef * over_j

            reward = qoe_sum - penalty
            R_total += reward

            # ---- 存经验 + 学习 ----
            if t < nf - 1:
                # 先更新 buffer 得到 s'
                bf_next = np.zeros(Nu)
                for i in range(Nu):
                    bf_next[i] = max(0.0, min(D_bf,
                        buffer_state[i, t]
                        + (download_rates[i, t] - r_final[i]) * dt))
                s_next = _build_state(Nu, bf_next, download_rates[:, t + 1],
                                      r_final, task_assignment[:, t + 1],
                                      D_bf, R_max, chi_star)
                done_flag = (t == nf - 2)
            else:
                s_next = s
                done_flag = True

            if train_mode:
                agent.store(s, a, reward, s_next, done_flag)
                agent.update()

        return R_total, r_decision, buffer_state

    # ---- 训练 ----
    for ep in range(num_episodes):
        R, _, _ = run_one_episode(train_mode=True)
        agent.decay_noise()
        episode_rewards.append(R)
        if verbose and ep % 10 == 0:
            print(f"[DDPG] ep={ep:4d}  R={R:8.2f}  noise={agent.noise_sigma:.3f}")

    # ---- 评估(关噪声) ----
    # 计算纯 QoE (不扣 penalty),与 KKT 直接可比
    _, r_decision_eval, buffer_eval = run_one_episode(train_mode=False)

    wait_cur = initial_wait_times.copy()
    eval_qoe = 0.0
    r_previous_eval = np.zeros((Nu, nf))
    for i in range(Nu):
        r_previous_eval[i, 0] = r_decision_eval[i, 0]
    for t in range(nf):
        if t > 0:
            r_previous_eval[:, t] = r_decision_eval[:, t - 1]
            wait_cur[:] = 0.0
        for i in range(Nu):
            eval_qoe += _qoe_new(buffer_eval[i, t], r_decision_eval[i, t],
                                 r_previous_eval[i, t], wait_cur[i],
                                 w1, w2, w3, w4, dt, xi)

    return agent, episode_rewards, eval_qoe, r_decision_eval, buffer_eval
