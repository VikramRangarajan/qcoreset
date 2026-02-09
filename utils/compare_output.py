from itertools import combinations
import numpy as np


def compute_ratio_err(pred_list, label):
    # diversity metrics 1. ratio-error
    ratio_err = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i, j = comb
        err_i = (np.argmax(pred_list[i], axis=-1) != np.argmax(label, axis=-1)).astype(
            np.float32
        )
        err_j = (np.argmax(pred_list[j], axis=-1) != np.argmax(label, axis=-1)).astype(
            np.float32
        )
        same = (err_i * err_j).sum()
        diff = (1.0 - (1.0 - err_i) * (1.0 - err_j)).sum() - same
        ratio_err.append(diff / same)
    ratio_err = np.mean(ratio_err)
    return ratio_err


def compute_q_stat(pred_list, label):
    # diversity metrics 2. q-stat
    q_stat = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i, j = comb
        err_i = (np.argmax(pred_list[i], axis=-1) != np.argmax(label, axis=-1)).astype(
            np.float32
        )
        err_j = (np.argmax(pred_list[j], axis=-1) != np.argmax(label, axis=-1)).astype(
            np.float32
        )
        n_00 = (err_i * err_j).sum()
        n_01 = (err_i * (1.0 - err_j)).sum()
        n_10 = ((1.0 - err_i) * err_j).sum()
        n_11 = ((1.0 - err_i) * (1.0 - err_j)).sum()
        q_stat.append((n_11 * n_00 - n_01 * n_10) / (n_11 * n_00 + n_01 * n_10))
    q_stat = np.mean(q_stat)
    return q_stat


def compute_cc(pred_list, label):
    # diversity metrics 3. correlation coefficient
    cc = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i, j = comb
        err_i = (np.argmax(pred_list[i], axis=-1) != np.argmax(label, axis=-1)).astype(
            np.float32
        )
        err_j = (np.argmax(pred_list[j], axis=-1) != np.argmax(label, axis=-1)).astype(
            np.float32
        )
        cc.append(np.corrcoef(err_i, err_j))
    cc = np.mean(cc)
    return cc


def compute_disagree(pred_list, label):
    # diversity metric 4. disagree
    disagree = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i, j = comb
        dis = (
            (np.argmax(pred_list[i], axis=1) != np.argmax(pred_list[j], axis=1))
            .astype(np.float32)
            .mean()
        )
        disagree.append(dis)
    disagree = np.mean(disagree)
    return disagree
