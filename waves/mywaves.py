import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyWavelet:
    def __init__(self, data,wavelet, levels = 0):
        self.data = data
        self.wavelet = wavelet
        if levels == 0:
            self.levels = self._levels()
        else:
            self.levels = levels
    def _levels(self):
        n = 1
        max_levels = 0
        data_len = self.data.shape[0]
        while True:
            old_n = n
            n = n*2
            if data_len - n < 0:
                n = old_n
                break
            max_levels+=1
        return max_levels

    def _signal_decomp(self):
        """
        wavelet = vilket typ av wavelet??
        levels = ??
        waverec är inverse transform
        pywt returnar approximation coefs och detail coefs
        Returns:
            rec_a <- 
        """
        w = pywt.Wavelet(self.wavelet)
        mode = pywt.Modes.constant

        ca = []
        cd = []
        for i in range(self.levels):
            (a,d) = pywt.dwt(self.data, w, mode)
            ca.append(a)
            cd.append(d)
        rec_a = []
        rec_d = []
        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None]*i
            rec = pywt.waverec(coeff_list, w, mode)
            rec_a.append(rec)
        for i, coeff in enumerate(cd):
            coeff_list = [coeff, None] + [None]*i
            rec = pywt.waverec(coeff_list, w, mode)
            rec_d.append(rec)
        return rec_a, rec_d
    def mra(self):
        """
        vad fan gör denna
        """
        rec_a, rec_d = self._signal_decomp()
        v = rec_a[-1].squeeze()
        n = self.data.shape[0]
        extra_n = len(v) - n
        v = v[:-extra_n]
        w_j = np.zeros((len(v), len(rec_d)))
        for i in range(len(rec_d)):
            if len(rec_d[i]) > n:
                extra_n = len(rec_d[i])-n
                w_j[:, i] = rec_d[i].squeeze()[:-extra_n]
            else:
                w_j[:, i] = rec_d[i].squeeze()
        return v, w_j
    def signals_low_freq(self, w_j, n_days=60):
        d = pd.DataFrame(w_j[:,8:12].sum(axis=1), index=self.data.index)
        d_diff = d.diff(periods=1)
        days = 0
        signals_dict = {}
        for i in range(2, len(d_diff)):
            if np.sign(d_diff.iloc[i-1][0]) != np.sign(d_diff.iloc[i][0]): #crossing zero
                days = 1
            else: #same side
                days += 1
                if days > n_days:
                    signals_dict[d_diff.index[i]] = np.sign(d_diff.iloc[i][0])
                else:
                    signals_dict[d_diff.index[i]] = 0
        return signals_dict
    def signals_high_freq(self, w_j, zscore_thresh=2.0):
        d = pd.DataFrame(w_j[:, 0:1].sum(axis=1), index=self.data.index)
        rolling = d.rolling(252)
        zscore = (d - rolling.mean())/rolling.std().dropna()
        d = d.loc[zscore.index]
        signals_dict = {}
        curr_signal = 0
        for i in range(len(zscore)):
            score = zscore.iloc[i][0]
            s = np.sign(score)
            if abs(score) > zscore_thresh: #signal
                curr_signal = -s
                signals_dict[zscore.index[i]] = curr_signal
            elif curr_signal != 0: #check if exit is met ??
                if curr_signal == 1:
                    if score > 1: #exit
                        curr_signal = 0
                elif curr_signal == -1:
                    if score < -1:
                        curr_signal = 0
            signals_dict[zscore.index[i]] = curr_signal
        return signals_dict
    def _zero_padd_signals(self, signals_df):
        dd = {d: (signals_df.loc[d][0] if d in signals_df.index else 0.0) for d in self.data.index}
        return dd
    def combined_signals(self, w_j, n_days=60, zscore_thresh = 2.0):
        hf_dict = self.signals_high_freq(w_j, zscore_thresh)
        hf_df = pd.DataFrame(hf_dict, index=["High Freq"]).T
        hf_padded = pd.DataFrame(self._zero_padd_signals(hf_df), index=["High Freq"]).T

        lf_dict = self.signals_low_freq(w_j, n_days)
        lf_df = pd.DataFrame(lf_dict, index=["Low Freq"]).T
        lf_padded = pd.DataFrame(self._zero_padd_signals(lf_df), index=["Low Freq"]).T
        
        signals_dict = {}
        w1 = 0.5*2
        w2 = (1-w1)*2
        comb_signals_df = pd.DataFrame(lf_padded.values * w1 + hf_padded.values*w2, index=lf_padded.index)
        return dict(comb_signals_df)
    def entry_exit(self, signals_dict):
        signals_df = pd.DataFrame(signals_dict, index = ["Low Freq Signals"]).T
        signals_date = [signals_df.index[i] for i in range(1, len(signals_df)-1) if (signals_df.iloc[i-1][0] != signals_df.iloc[i][0])]
        buy_signals_date = [signals_date[i] for i in range(len(signals_date)) if signals_df.loc[signals_date[i]][0] == 1.0]
        sell_signals_date = [signals_date[i] for i in range(len(signals_date)) if signals_df.loc[signals_date[i]][0] == -1.0]
        exit_signals_date = [signals_date[i] for i in range(len(signals_date)) if signals_df.loc[signals_date[i]][0] == 0.0]
        return buy_signals_date, sell_signals_date, exit_signals_date
    def equity_curve(self, signals_dict):
        buy_date, sell_date, exit_date = self.entry_exit(signals_dict)
        rets = self.data.pct_change().dropna()
        signals = []
        for i in rets.index:
            if i in buy_date:
                signals.append(1)
            elif i in sell_date:
                signals.append(-1)
            else:
                signals.append(0)
        r = rets.values*signals
        ec = (1.0 + r).cumprod()
        plt.plot(ec)

        plt.show()


    ### -- PLOTTING -- ###
    def plot_mra(self):
        v, w_j = self.mra()
        data = self.data
        f, axarr = plt.subplots(w_j.shape[1]+2, sharex=True)
        axarr[0].plot(data.index, data.values.squeeze())
        for i in range(w_j.shape[1]):
            axarr[i+1].plot(data.index, w_j[:,i])
        axarr[i+2].plot(data.index, v)
        plt.show()
    def plot_signals_low_freq(self, v, w_j, n_days = 60):
        signals_dict = self.signals_low_freq(w_j, n_days)
        buy_date, sell_date, exit_date = self.entry_exit(signals_dict)
        
        plt.plot(self.data.index, self.data.values.squeeze(), label = "Underlying", linestyle="-.", color="gray", alpha=0.2)
        plt.scatter(pd.DatetimeIndex(buy_date), self.data.loc[buy_date].squeeze(), marker="^", color="g")
        plt.scatter(pd.DatetimeIndex(sell_date), self.data.loc[sell_date], marker="v", color="r")
        plt.scatter(pd.DatetimeIndex(exit_date), self.data.loc[exit_date], marker="o", color="b")
        plt.show()
    def plot_signals_high_freq(self, v, w_j):
        signals_dict = self.signals_high_freq(w_j)
        buy_date, sell_date, exit_date = self.entry_exit(signals_dict)
        plt.plot(self.data.index, self.data.values.squeeze(), label = "Underlying", linestyle="-.", color="gray", alpha=0.2)
        plt.scatter(pd.DatetimeIndex(buy_date), self.data.loc[buy_date].squeeze(), marker="^", color="g")
        plt.scatter(pd.DatetimeIndex(sell_date), self.data.loc[sell_date], marker="v", color="r")
        plt.scatter(pd.DatetimeIndex(exit_date), self.data.loc[exit_date], marker="o", color="b")
        plt.show()
    def plot_signals_combined(self, v, w_j):
        signals_dict = self.combined_signals(w_j)
        buy_date, sell_date, exit_date = self.entry_exit(signals_dict)
        print(buy_date)
        plt.plot(self.data.index, self.data.values.squeeze(), label = "Underlying", linestyle="-.", color="gray", alpha=0.2)
        plt.scatter(pd.DatetimeIndex(buy_date), self.data.loc[buy_date].squeeze(), marker="^", color="g")
        plt.scatter(pd.DatetimeIndex(sell_date), self.data.loc[sell_date], marker="v", color="r")
        plt.scatter(pd.DatetimeIndex(exit_date), self.data.loc[exit_date], marker="o", color="b")
        plt.show()

