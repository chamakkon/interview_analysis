import numpy as np


def scalerize_seq_data(sequence_data, segment_sequence):
    """
    時系列データを 4 つのスカラー特徴量に変換する。

    1. 全区間の平均値
    2. トレンド（時間に対する線形回帰の傾き）
    3. RMSSD（Root Mean Square of Successive Differences）
    4. seg3 の平均 - seg1 の平均

    Parameters
    ----------
    sequence_data : array-like, shape (T,)
        時系列の値
    segment_sequence : array-like, shape (T,)
        各時刻が属するセグメント名（例: "seg1", "seg2", "seg3", ...）

    Returns
    -------
    mean_val : float
    trend_slope : float
    rmssd : float
    seg3_minus_seg1 : float
    """
    x = np.asarray(sequence_data, dtype=float)
    seg = np.asarray(segment_sequence)

    if x.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    # 1. 平均
    mean_val = float(np.nanmean(x))

    # 2. トレンド（線形回帰の傾き）
    if x.size >= 2:
        t = np.arange(x.size, dtype=float)
        # 欠損値があれば落として回帰
        mask = ~np.isnan(x)
        if mask.sum() >= 2:
            slope, _ = np.polyfit(t[mask], x[mask], 1)
            trend_slope = float(slope)
        else:
            trend_slope = np.nan
    else:
        trend_slope = np.nan

    # 3. RMSSD
    if x.size >= 2:
        diff = np.diff(x)
        # diff に NaN がある場合は無視
        valid = ~np.isnan(diff)
        if valid.any():
            rmssd = float(np.sqrt(np.mean(diff[valid] ** 2)))
        else:
            rmssd = np.nan
    else:
        rmssd = np.nan

    # 4. seg3 の平均 - seg1 の平均
    seg1_mask = seg == "seg1"
    seg3_mask = seg == "seg3"

    if seg1_mask.any():
        seg1_mean = float(np.nanmean(x[seg1_mask]))
    else:
        seg1_mean = np.nan

    if seg3_mask.any():
        seg3_mean = float(np.nanmean(x[seg3_mask]))
    else:
        seg3_mean = np.nan

    seg3_minus_seg1 = (
        float(seg3_mean - seg1_mean)
        if not (np.isnan(seg1_mean) or np.isnan(seg3_mean))
        else np.nan
    )

    return mean_val, trend_slope, rmssd, seg3_minus_seg1


