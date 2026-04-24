# -*- coding: utf-8 -*-
"""
hit_rate.py
===========
对齐 cal_hit_rate.m:
    请求级命中率:每个用户请求一个视频,判断能否被 IU 或 SBS 命中(t=0 位置判定)。
    IU 命中优先于 SBS 命中。
"""
import numpy as np
import config as cfg


def compute_hit_rate(
    topo, iu_indices, iu_flags, cache_decision,
    community_users, requested_videos,
    target_community=cfg.TARGET_COMMUNITY,
):
    m0 = target_community - 1
    Nu = len(community_users)
    iu_per_comm = cfg.IU_PER_COMMUNITY
    total_users = cfg.TOTAL_USERS
    target_iu_row = iu_indices[m0]

    iu_hit_total = 0
    sbs_hit_total = 0
    for i, uid in enumerate(community_users):
        req = requested_videos[i]

        # ---- IU 命中 ----
        iu_hit = False
        if iu_flags[uid] == 1 and cache_decision[uid, req] == 1:
            iu_hit = True
        else:
            for j in range(iu_per_comm):
                iu_idx = int(target_iu_row[j])
                if cache_decision[iu_idx, req] == 1:
                    diff = (topo.now_user_positions[uid, :, 0]
                            - topo.now_user_positions[iu_idx, :, 0])
                    if np.linalg.norm(diff) <= cfg.IU_COVERAGE:
                        iu_hit = True
                        break

        # ---- SBS 命中 ----
        sbs_hit = False
        if (not iu_hit) and cache_decision[total_users + m0, req] == 1:
            sbs_hit = True

        if iu_hit:
            iu_hit_total += 1
        elif sbs_hit:
            sbs_hit_total += 1

    total = Nu
    return (
        (iu_hit_total + sbs_hit_total) / total,
        iu_hit_total / total,
        sbs_hit_total / total,
    )
