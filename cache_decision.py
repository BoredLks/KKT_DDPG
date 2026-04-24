# -*- coding: utf-8 -*-
"""
cache_decision.py
=================
完整对齐 cal_cache_decision_MOIS.m 的两阶段放置。

Stage A: SBS 缓存 (跨 SBS 无重复, 公式约束 34c)
  - 对每个 SBS m,把 fused_preference[N+m, :] 里已被其他 SBS 缓存的文件置 -inf
  - 按偏好降序,正偏好且容量够的放入
  - 更新 cached_files_by_sbs

Stage B: IU 缓存 (IU 间可重复)
  - 对每个 IU, 按 fused_preference[iu_idx, :] 降序
  - 正偏好且容量够的放入
"""
import numpy as np
import config as cfg


def cal_cache_decision_mois(fused_preference, iu_indices,
                             D_eg=cfg.D_EG, D_iu=cfg.D_IU,
                             nf=cfg.N_F, v_chi_star=cfg.V_CHI_STAR):
    N = cfg.TOTAL_USERS
    H = cfg.H
    F = cfg.F_FILES
    iu_per_comm = cfg.IU_PER_COMMUNITY

    cache_decision = np.zeros((N + H, F), dtype=np.int64)
    file_size = nf * v_chi_star

    # ---- Stage A: SBS 缓存 ----
    cached_by_sbs = np.zeros(F, dtype=bool)
    sbs_cache_space = np.zeros(H)
    for m in range(H):
        sbs_idx = N + m
        prefs = fused_preference[sbs_idx].copy()
        prefs[cached_by_sbs] = -np.inf
        order = np.argsort(-prefs)
        for f in order:
            if prefs[f] <= 0:
                break
            if sbs_cache_space[m] + file_size <= D_eg:
                cache_decision[sbs_idx, f] = 1
                sbs_cache_space[m] += file_size
                cached_by_sbs[f] = True
            else:
                break

    # ---- Stage B: IU 缓存 ----
    iu_cache_space = np.zeros(N)
    for m in range(H):
        for j in range(iu_per_comm):
            iu_idx = int(iu_indices[m, j])
            prefs = fused_preference[iu_idx]
            order = np.argsort(-prefs)
            for f in order:
                if prefs[f] <= 0:
                    break
                if iu_cache_space[iu_idx] + file_size <= D_iu:
                    cache_decision[iu_idx, f] = 1
                    iu_cache_space[iu_idx] += file_size
                else:
                    break

    return cache_decision
