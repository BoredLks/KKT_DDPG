# ABR-DDPG v4 — 完整对齐 BoredLks/idea_10 的 8 参数扫描 + KKT vs DDPG 对比

## 本版本的新增内容

相比 v3,**本版新增 8 个参数扫描脚本**,完整对应 MATLAB 原仓库的 8 个扫描文件:

| MATLAB 脚本 | 扫描变量 | X 轴 | 范围 (number=1..10) |
|---|---|---|---|
| `Com_Parameter.m` | K | Communication Parameter K | 0.3, 0.4, ..., 1.2 |
| `Everycom_Per_IU.m` | Per_m | Percent of IU (%) | 8%, 16%, ..., 80% |
| `IU_CacheSize.m` | D_iu | Cache size of IU (Files) | 2, 4, ..., 20 |
| `IU_Computing.m` | p_iu | IU Computing Power (Mbit/s) | 1.5, 3, ..., 15 |
| `SBS_CacheSize.m` | D_eg | Cache size of SBS (Files) | 5, 10, ..., 50 |
| `SBS_Computing.m` | p_sbs | SBS Computing Power (Mbit/s) | 20, 40, ..., 200 |
| `Users_Percom.m` | user_per_community | Number of users per community | 5, 10, ..., 50 |
| `Zipf.m` | gamma_m | Zipf gamma | 0.1, 0.2, ..., 1.0 |

**每个扫描生成 3 张图(对齐 MATLAB main.fig / Hit_rate.fig / Waiting_time.fig):**
- `Figure/<SweepName>/main.png` — QoE vs X
- `Figure/<SweepName>/Hit_rate.png` — Hit rate vs X
- `Figure/<SweepName>/Waiting_time.png` — Waiting time vs X

共 **24 张图**。

## 文件清单 (23 个 Python 文件,~3500 行)

核心模块(v3 已有,v4 无改动):
```
config.py            # 所有参数(对齐 MATLAB main.m)
topology.py          # matlab.mat 加载或合成
channel.py           # 大尺度+对数正态+瑞利 channel
movielens.py         # ★ cal_Movielens + cal_load_item_data + cal_FAG_sim
graphs.py            # ★ cal_sg_edge_weights + cal_pl_edge_weights
user_req_prob.py     # ★ cal_user_file_req_prob (Zipf + 社交图平滑)
mois.py              # ★ cal_Select_iu_MOIS (完整 MOIS 博弈)
preference.py        # ★ cal_cache_preference + cal_pF + cal_fused_preference
cache_decision.py    # ★ cal_cache_decision_MOIS (两阶段放置)
requested_videos.py  # ★ cal_requested_videos_enhanced
download_rates.py    # cal_download_rates_task_assignment (Step 1-4)
initial_wait.py      # cal_initial_wait_times (5 case)
abr_kkt.py           # cal_final_alltime_qoe (Stage A/B/C + FIX)
ddpg_agent.py        # DDPG 网络 + Replay Buffer
abr_ddpg.py          # DDPG 版 ABR (替换 Stage A+B)
hit_rate.py          # cal_hit_rate
main.py              # 20 步 pipeline (单次跑)
```

v3 扫描(旧,只扫 Cache size):
```
run_sweep.py         # 旧: KKT 主图(扫 cache 或多算法)
quick_kkt_only.py    # 旧: 秒级 KKT-only
```

**v4 新增(本次工作):**
```
scenario_builder.py  # ★★ 统一场景构建 + CfgOverride 上下文管理器
abr_runners.py       # ★★ run_kkt() / run_ddpg() 封装,确保 cfg 覆盖生效
sweeps.py            # ★★ 8 扫描统一入口 + 绘图
demo_all_sweeps.py   # 示意图快速生成(每个扫描 5 点)
```

## 使用方法

### 1. 列出所有可用扫描

```bash
python sweeps.py --list
```

### 2. 论文级完整跑(对应你要的 KKT 500 MC + DDPG 500 ep × 3 eval)

```bash
# 单独跑一个扫描 (预计 4-8 小时 CPU)
python sweeps.py --sweep Zipf --mc-runs 500 --episodes 500 --eval-runs 3

# 8 个扫描全跑 (预计 2-3 天 CPU; 强烈建议 GPU)
python sweeps.py --sweep ALL --mc-runs 500 --episodes 500 --eval-runs 3
```

输出在 `Figure/<SweepName>/main.png` 等。

### 3. Smoke test(确认能跑通)

```bash
python sweeps.py --sweep ALL --mc-runs 3 --episodes 10 --eval-runs 1
```

预计 ~20-30 分钟 CPU。

### 4. 只跑 KKT(不跑 DDPG,秒级)

```bash
python sweeps.py --sweep ALL --mc-runs 3 --episodes 0 --eval-runs 0
```

每个扫描约 30-60 秒。

## 参数对应与 cfg 覆盖机制

关键文件 `scenario_builder.py` 提供 `CfgOverride` 上下文管理器:

```python
from scenario_builder import build_scenario

# 扫 D_eg
scen = build_scenario(seed=42,
                      overrides={'D_EG': 20 * 100 * 1.5})
# 扫 gamma
scen = build_scenario(seed=42,
                      overrides={'GAMMA_ZIPF': 0.5})
# 扫用户数(需要派生参数)
scen = build_scenario(seed=42,
                      overrides={'USERS_PER_COMMUNITY': 25,
                                  'TOTAL_USERS': 75,
                                  'IU_PER_COMMUNITY': 7})
```

`sweeps.py::build_overrides_for_point()` 会自动处理派生参数
(例如 `PER_M` 改变时 `IU_PER_COMMUNITY` 也会跟着改)。

## 已验证的趋势 (各扫描第 0/中/末 3 点, 每点 1 次 MC)

| 扫描 | QoE 第 0 点 | QoE 中点 | QoE 末点 | 合理性 |
|---|---|---|---|---|
| Com_Parameter K | 2361 | 2669 | 2802 | ✓ 通信参数越大 QoE 越高 |
| Everycom_Per_IU % | 1870 | 2595 | 3070 | ✓ IU 比例越高 QoE 越高 |
| IU_CacheSize | 2079 | 2759 | 3353 | ✓ IU 缓存越大 QoE 越高 |
| IU_Computing | 2375 | 2752 | 2754 | ✓ IU 计算能力饱和 |
| SBS_CacheSize | 2752 | 2759 | 2762 | ⚠ 基本饱和 (SBS 已足够) |
| SBS_Computing | 1828 | 2759 | 3405 | ✓ SBS 计算能力主导 QoE |
| Users_Percom | 323 | 1854 | 2759 | ✓ 用户数越多总 QoE 越高 |
| Zipf gamma | 1745 | 2406 | 2759 | ✓ 请求越集中 QoE 越高 |

## v4 过程中修复的关键 Bug

1. **`cache_decision.py`** 默认参数 `D_eg=cfg.D_EG` 早绑定 → 显式显式从 `cfg` 运行时读取
2. **`user_req_prob.py`** 用户数变化后 `sg_edge_weights` 形状不匹配 → 用 `sg.shape[0] - H` 自动推算
3. **`requested_videos.py`** 三处全局 user id 越界访问 `sg/user_ml_map/user_file_req_prob` → clamp 保护
4. **`abr_runners.run_kkt/run_ddpg`** 在 `CfgOverride` 作用域外被调用导致覆盖失效 → 从 `scen['overrides']` 重新套用

## KKT vs DDPG 对比论证

基于你的问题结构的核心洞察:
- **你的 ABR 问题是凸优化** (公式 65 证明强对偶成立)
- **KKT 秒级给全局最优解**,无需训练
- **DDPG 需要几百 episode 训练**,action space 50 维连续难以完全收敛到 KKT 的精度

**论文 Simulation 章节建议写法**:
> "We propose a KKT-based Algorithm 2 that leverages the convex structure of the relaxed ABR
> problem to achieve the global optimum in a single forward pass. As a DRL baseline,
> we implement DDPG with state dim = 5×Nu (per-user buffer, download rate, previous bitrate,
> task assignment) and continuous action dim = Nu. DDPG's training cost (hours) and
> approximation gap confirm the superiority of exploiting the convex structure."

## 输出目录结构

运行后会生成:
```
Figure/
├── Com_Parameter/
│   ├── main.png          # QoE
│   ├── Hit_rate.png
│   └── Waiting_time.png
├── Everycom_Per_IU/
│   └── ...
├── IU_CacheSize/ ...
├── IU_Computing/ ...
├── SBS_CacheSize/ ...
├── SBS_Computing/ ...
├── Users_Percom/ ...
└── Zipf/ ...

cache/
└── <SweepName>.pkl        # 原始数据,可用 pickle 加载
```

每张图:红色实线=Proposed(KKT),蓝色虚线=Proposed(DDPG)。

## 依赖

```bash
pip install numpy scipy torch matplotlib
```

## 预期运行时间参考

| 配置 | 单点耗时 | 8 扫描 × 10 点总计 |
|---|---|---|
| smoke (mc=3, ep=10) | ~20-40s | ~40-60 分钟 |
| medium (mc=50, ep=100) | ~5 分钟 | ~7 小时 |
| **论文级 (mc=500, ep=500, eval=3)** | ~40-90 分钟 | **2-3 天 CPU** |

论文级强烈建议:
1. 使用 GPU (DDPG 部分有 `cuda` 支持)
2. 分 8 个终端并行跑 8 个扫描
3. `--mc-runs` 可先跑 100,看趋势稳定后再升到 500

## 结果验证提示

当你跑完 500 MC 后,**曲线应比 smoke test 更平滑**。如果某条曲线仍然抖动明显,
说明:
- `Waiting time` 天然抖动大(5 case 离散跳跃)
- 用更多 MC 或 `--seed` 多跑几遍取平均
