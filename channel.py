# -*- coding: utf-8 -*-
"""
channel.py
==========
信道模型:大尺度路径损耗 + 对数正态阴影 + 瑞利小尺度。
严格对齐 MATLAB cal_download_rates_task_assignment.m:
    vartheta = lognrnd(slowfading_avg, slowfading_sd)
    re = (randn() + 1i*randn()) / sqrt(2)
    xi = abs(re)^2
    channel_gain = K * vartheta * xi * dist^(-epsilon)
    rate = B * log2(1 + P*gain / (N0 + interference))
"""
import numpy as np
import config as cfg


def slow_fading_params():
    """dB → 对数标准差与均值 (MATLAB L453-454)."""
    sd = cfg.SLOW_FADING_DB / (10.0 * np.log10(np.e))
    avg = -sd ** 2 / 2.0
    return avg, sd


def channel_gain(dist, rng):
    """单次采样信道增益: K * vartheta * xi * dist^(-eps)."""
    if dist <= 0:
        return 0.0
    avg, sd = slow_fading_params()
    vartheta = rng.lognormal(avg, sd)
    re_r = rng.randn()
    re_i = rng.randn()
    re = (re_r + 1j * re_i) / np.sqrt(2)
    xi = abs(re) ** 2
    return cfg.K_CH * vartheta * xi * (dist ** (-cfg.EPS_PATHLOSS))


def sinr_rate(signal_power, interference_power, bw_mhz=cfg.B_MHZ):
    """Shannon 速率 (Mbps)."""
    denom = cfg.N0 + interference_power
    if denom <= 0:
        return 0.0
    sinr = signal_power / denom
    sinr = max(min(sinr, 1e12), 0.0)
    return bw_mhz * np.log2(1.0 + sinr)


def r_max_sbs():
    """最大理论速率 B*log2(1 + P_sbs*K/N0),MATLAB L450."""
    return cfg.B_MHZ * np.log2(1.0 + cfg.P_SBS * cfg.K_CH / cfg.N0)
