# -*- coding: utf-8 -*-
"""
movielens.py
============
完整对齐 cal_Movielens.m 和 cal_load_item_data.m:

cal_Movielens:
    读取 u.data (user_id, movie_id, rating, timestamp)
    构造评分矩阵 R,做行单位向量归一化
    用户间余弦相似度 sim_matrix = R_norm @ R_norm.T

cal_load_item_data:
    读取 u.item,解析 19 列 genre one-hot 与发行年份
    返回 (genre_matrix_ml, release_years_ml, num_genres=19)
"""
import os
import numpy as np
import config as cfg


# ------------------------------------------------------------
def load_udata(file_path="ml-100k/u.data"):
    """读取 u.data,返回 (num_users, num_movies) 评分矩阵 R."""
    data = np.loadtxt(file_path, dtype=np.int64)
    user_ids = data[:, 0]
    movie_ids = data[:, 1]
    ratings = data[:, 2]
    num_users = int(user_ids.max())
    num_movies = int(movie_ids.max())
    R = np.zeros((num_users, num_movies), dtype=np.float64)
    for uid, mid, r in zip(user_ids, movie_ids, ratings):
        R[uid - 1, mid - 1] = r          # MATLAB 1-based → Python 0-based
    return R, num_users, num_movies


# ------------------------------------------------------------
def cal_movielens(R):
    """余弦相似度矩阵 (num_users, num_users),对齐 cal_Movielens.m."""
    row_norm = np.sqrt((R ** 2).sum(axis=1, keepdims=True))
    row_norm[row_norm == 0] = 1.0
    R_norm = R / row_norm
    sim_matrix = R_norm @ R_norm.T
    return sim_matrix


# ------------------------------------------------------------
def cal_load_item_data(item_file_path="ml-100k/u.item", num_movies=1682):
    """
    解析 u.item:每行 | 分隔,格式为
        movie_id | title | release_date | video_release | url | genre_0..genre_18
    第 6..24 列是 19 个 genre 的 one-hot。
    """
    num_genres = 19
    genre_matrix_ml = np.zeros((num_movies, num_genres), dtype=np.int64)
    release_years_ml = np.zeros(num_movies, dtype=np.int64)

    if not os.path.exists(item_file_path):
        # 回退为随机属性
        rng = np.random.RandomState(0)
        genre_matrix_ml = rng.randint(0, 2, size=(num_movies, num_genres))
        release_years_ml = rng.randint(1920, 1999, size=num_movies)
        return genre_matrix_ml, release_years_ml, num_genres

    with open(item_file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 5 + num_genres:
                continue
            try:
                mid = int(parts[0]) - 1               # 0-based
            except ValueError:
                continue
            if mid >= num_movies:
                continue
            # genre one-hot (第 6..24 列,即 parts[5:5+19])
            for g in range(num_genres):
                try:
                    genre_matrix_ml[mid, g] = int(parts[5 + g])
                except (ValueError, IndexError):
                    pass
            # 发行日期 (第 3 列,格式 "DD-MMM-YYYY")
            ds = parts[2].strip()
            if ds:
                tokens = ds.split('-')
                if len(tokens) >= 3:
                    try:
                        release_years_ml[mid] = int(tokens[2])
                    except ValueError:
                        pass
    return genre_matrix_ml, release_years_ml, num_genres


# ------------------------------------------------------------
def cal_fag_sim(F, num_movies, num_genres, genre_matrix_ml, release_years_ml,
                file_movielens_mapping):
    """
    完整对齐 cal_FAG_sim.m,构造 FAG 文件相似矩阵 (F, F)。
    - 属性权重 omega_k = Count(k) / sum(Count(r))   (公式 17)
    - 文件间边权 = Σ 属性匹配的权重                   (公式 15, 16)
    """
    # ---- 1) 映射每个系统文件到 MovieLens 属性 ----
    file_genre_matrix = np.zeros((F, num_genres), dtype=np.int64)
    file_release_decade = np.zeros(F, dtype=np.int64)
    for k in range(F):
        ml_id = file_movielens_mapping[k]
        if ml_id < genre_matrix_ml.shape[0]:
            file_genre_matrix[k] = genre_matrix_ml[ml_id]
        if ml_id < release_years_ml.shape[0] and release_years_ml[ml_id] > 0:
            file_release_decade[k] = (release_years_ml[ml_id] // 10) * 10

    # ---- 2) 公式 (17) 属性权重 ----
    A_total = num_genres + 1          # 19 genre + 1 decade
    count_attr = np.zeros(A_total, dtype=np.float64)
    for g in range(num_genres):
        count_attr[g] = file_genre_matrix[:, g].sum()
    valid = file_release_decade[file_release_decade > 0]
    if valid.size > 0:
        uniq, counts = np.unique(valid, return_counts=True)
        count_attr[-1] = counts.max()
    else:
        count_attr[-1] = 1.0

    s = count_attr.sum()
    if s > 0:
        omega_attr = count_attr / s
    else:
        omega_attr = np.ones(A_total) / A_total

    # ---- 3) 公式 (15)(16) 边权 ----
    FAG_sim = np.zeros((F, F), dtype=np.float64)
    for fi in range(F):
        for fj in range(fi + 1, F):
            w = 0.0
            for g in range(num_genres):
                if file_genre_matrix[fi, g] == 1 and file_genre_matrix[fj, g] == 1:
                    w += omega_attr[g]
            if (file_release_decade[fi] > 0 and
                    file_release_decade[fj] > 0 and
                    file_release_decade[fi] == file_release_decade[fj]):
                w += omega_attr[-1]
            FAG_sim[fi, fj] = w
            FAG_sim[fj, fi] = w
    return FAG_sim
