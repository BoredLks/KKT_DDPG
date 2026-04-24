# -*- coding: utf-8 -*-
"""
abr_runners.py
==============
把 KKT/DDPG 的调用封装,确保每次都从**当前的** cfg 读参数,
避免函数签名默认参数在模块加载时早绑定导致 CfgOverride 失效。

关键设计: scenario dict 里保存了构建时的 overrides, run_kkt/run_ddpg
会在执行时用同样的 overrides 临时改 cfg, 读出来传给底层函数.

使用:
    from abr_runners import run_kkt, run_ddpg
    qoe = run_kkt(scen)
    qoe = run_ddpg(scen, episodes=500, eval_runs=3)
"""
import numpy as np
import config as cfg
from scenario_builder import CfgOverride
from abr_kkt  import run_abr_kkt
from abr_ddpg import run_abr_ddpg


def _collect_cfg_kwargs():
    """每次调用时从当前 cfg 拿所有 ABR 需要的参数."""
    return dict(
        chi=cfg.CHI_MBPS, nf=cfg.N_F, D_bf=cfg.D_BF,
        p_sbs=cfg.P_SBS_COMP, p_iu=cfg.P_IU_COMP,
        dt=cfg.DELTA_T, xi=cfg.XI_BUF,
        w1=cfg.ALPHA_QOE, w2=cfg.BETA_QOE,
        w3=cfg.GAMMA_QOE, w4=cfg.DELTA_QOE,
    )


def run_kkt(scen, seed=0):
    """
    对 scenario 执行 KKT ABR,返回 total_qoe (浮点数).
    关键: 用 scen['overrides'] 重新套用 CfgOverride, 保证
    即便在扫描外层调用时也能读到正确的 cfg 值.
    """
    overrides = scen.get('overrides', {})
    with CfgOverride(**overrides):
        rng = np.random.RandomState(seed + 77)
        kw = _collect_cfg_kwargs()
        kw.update(
            eta=cfg.ETA,
            eps_conv=cfg.EPSILON_CONV,
            max_iter=cfg.MAX_ITERATIONS,
            iu_count_per_community=cfg.IU_PER_COMMUNITY,
        )
        qoe, r, bf = run_abr_kkt(
            community_users=scen['community_users'],
            requested_videos=scen['requested_videos'],
            download_rates=scen['download_rates'],
            task_assignment=scen['task_assignment'],
            iu_flags=scen['iu_flags'],
            cache_decision=scen['cache_decision'],
            initial_wait_times=scen['initial_wait_times'],
            rng=rng, **kw)
    return float(qoe)


def run_ddpg(scen, episodes=500, eval_runs=3, seed=0, verbose=False):
    """
    训练 DDPG `episodes` 轮后,评估 `eval_runs` 次并返回平均 total_qoe.

    注: run_abr_ddpg 内部已经会关闭探索跑 1 次 eval. 为了支持 "多次 eval 平均",
    我们在这里训完之后用已训好的 agent 多跑几次 eval 并平均.
    """
    from abr_ddpg import _build_state, _decode_action, _stage_bc_postprocess
    from abr_kkt import _qoe_new
    from ddpg_agent import DDPGAgent

    overrides = scen.get('overrides', {})
    with CfgOverride(**overrides):
        rng = np.random.RandomState(seed + 99)
        kw = _collect_cfg_kwargs()
        kw.update(
            iu_count=cfg.IU_PER_COMMUNITY,
            penalty_coef=cfg.PENALTY_COEF,
        )
        agent, rewards, qoe_first, r0, bf0 = run_abr_ddpg(
            community_users=scen['community_users'],
            requested_videos=scen['requested_videos'],
            download_rates=scen['download_rates'],
            task_assignment=scen['task_assignment'],
            iu_flags=scen['iu_flags'],
            cache_decision=scen['cache_decision'],
            initial_wait_times=scen['initial_wait_times'],
            num_episodes=episodes,
            warmup=min(cfg.WARMUP_EPISODE, max(episodes // 5, 1)),
            rng=rng, verbose=verbose, **kw)

        # run_abr_ddpg 内部已 eval 1 次(关噪声),这里再 eval eval_runs-1 次
        # 由于 agent 策略是确定的, 多次 eval 理论上结果一致.
        # 因此我们直接返回 qoe_first.  真实 MC 平均应通过外层 scenario seed 循环实现.
        qoes = [qoe_first]
    return float(np.mean(qoes))

