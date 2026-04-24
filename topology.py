# -*- coding: utf-8 -*-
"""
topology.py
===========
从 matlab.mat 加载固定位置数据,与 MATLAB main.m L93-98 完全对齐:
    sbs_positions      (3, 2)
    per_user_positions (150, 2, 100)   用于 MOIS IU 选择期间(大时间尺度内的平均轨迹)
    now_user_positions (150, 2, 100)   用于真实 ABR 阶段的逐时隙位置
    user_community     (150, 1)

如果找不到 .mat,就按 MATLAB 的生成逻辑合成一份(PPP 近似:社区内极坐标均匀)。
"""
import os
import numpy as np
import scipy.io as sio

import config as cfg


class Topology(object):
    def __init__(self, mat_path=None, seed=cfg.SEED_DEFAULT):
        self.rng = np.random.RandomState(seed)
        self.h = cfg.H
        self.users_per_comm = cfg.USERS_PER_COMMUNITY
        self.total_users = cfg.TOTAL_USERS
        self.T = cfg.T_SMALL

        if mat_path and os.path.exists(mat_path):
            self._load_from_mat(mat_path)
        else:
            self._synthesize()

    # -----------------------------------------------------
    def _load_from_mat(self, mat_path):
        d = sio.loadmat(mat_path, variable_names=[
            'sbs_positions', 'per_user_positions',
            'now_user_positions', 'user_community'
        ])
        self.sbs_positions      = np.asarray(d['sbs_positions'], dtype=float)
        self.per_user_positions = np.asarray(d['per_user_positions'], dtype=float)
        self.now_user_positions = np.asarray(d['now_user_positions'], dtype=float)
        uc = np.asarray(d['user_community']).flatten().astype(np.int64)
        # MATLAB 1-based -> 保持与 iu_indices 一致 (iu_indices 也是 1-based)
        self.user_community = uc

        # 调整时隙数以免 .mat 存的是 200 而 config 要 100
        if self.now_user_positions.shape[-1] < self.T:
            # 重复最后一帧填充
            pad = self.T - self.now_user_positions.shape[-1]
            self.now_user_positions = np.concatenate([
                self.now_user_positions,
                np.repeat(self.now_user_positions[:, :, -1:], pad, axis=-1)
            ], axis=-1)
        else:
            self.now_user_positions = self.now_user_positions[:, :, :self.T]

        if self.per_user_positions.shape[-1] < self.T:
            pad = self.T - self.per_user_positions.shape[-1]
            self.per_user_positions = np.concatenate([
                self.per_user_positions,
                np.repeat(self.per_user_positions[:, :, -1:], pad, axis=-1)
            ], axis=-1)
        else:
            self.per_user_positions = self.per_user_positions[:, :, :self.T]

    # -----------------------------------------------------
    def _synthesize(self):
        """无 .mat 文件时,用与 MATLAB 等价的生成流程合成位置."""
        # 3 个社区中心(三角形布局)
        center_dist = cfg.REGION_SIZE / 2.5
        centers = np.zeros((3, 2))
        for i in range(3):
            ang = 2 * np.pi * i / 3 + np.pi / 6
            centers[i] = [center_dist * np.cos(ang),
                          center_dist * np.sin(ang)]
        centers += cfg.REGION_SIZE / 2
        self.sbs_positions = centers

        # 用户在社区半径内极坐标均匀
        pos0 = np.zeros((self.total_users, 2))
        uc = np.zeros(self.total_users, dtype=np.int64)
        for m in range(self.h):
            for n in range(self.users_per_comm):
                idx = m * self.users_per_comm + n
                r = cfg.COMMUNITY_RADIUS * np.sqrt(self.rng.rand())
                ang = 2 * np.pi * self.rng.rand()
                pos0[idx] = centers[m] + r * np.array([np.cos(ang), np.sin(ang)])
                uc[idx] = m + 1            # 1-based
        self.user_community = uc

        # 沿时隙做随机游走,每步最大移动 MAX_MOVEMENT_DIST
        per = np.zeros((self.total_users, 2, self.T))
        now = np.zeros((self.total_users, 2, self.T))
        per[:, :, 0] = pos0
        now[:, :, 0] = pos0
        for t in range(1, self.T):
            for u in range(self.total_users):
                m = uc[u] - 1
                for _ in range(10):   # 最多尝试 10 次以确保不出社区
                    ang = 2 * np.pi * self.rng.rand()
                    d = cfg.MAX_MOVEMENT_DIST * self.rng.rand()
                    cand = per[u, :, t - 1] + d * np.array([np.cos(ang),
                                                             np.sin(ang)])
                    if np.linalg.norm(cand - centers[m]) <= cfg.COMMUNITY_RADIUS:
                        per[u, :, t] = cand
                        now[u, :, t] = cand
                        break
                else:
                    per[u, :, t] = per[u, :, t - 1]
                    now[u, :, t] = per[u, :, t - 1]

        self.per_user_positions = per
        self.now_user_positions = now

    # -----------------------------------------------------
    def community_members(self, community_1based):
        """返回目标社区用户的全局 0-based 索引(与 cal_target_community_users.m 对齐)."""
        m = community_1based
        start = (m - 1) * self.users_per_comm
        end   = m * self.users_per_comm
        return np.arange(start, end, dtype=np.int64)
