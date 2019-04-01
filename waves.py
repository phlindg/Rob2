import pywt
import bqplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def signal_decomp(data, w, levels):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)
    mode = pywt.Modes.constant
    a = data
    ca = []
    cd = []
    for i in range(levels):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w, mode=mode))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w, mode=mode))

    return rec_a, rec_d


def mra(data, wavelet, levels):
    rec_a, rec_d = signal_decomp(data.values.squeeze(), wavelet, levels)
    v = rec_a[-1].squeeze()
    n = data.shape[0]
    extra_n = len(v) - n
    v = v[:-extra_n]
    w_j = np.zeros((len(v), len(rec_d)))
    for i in range(len(rec_d)):
        if len(rec_d[i]) > n:
            extra_n = len(rec_d[i]) - n
            w_j[:, i] = rec_d[i].squeeze()[:-extra_n]
        else:
            w_j[:, i] = rec_d[i].squeeze()
    return v, w_j

def plot_mra(data, v, w_j):
    f, axarr = plt.subplots(w_j.shape[1]+2, sharex=True)
    axarr[0].plot(data.index, data.values.squeeze())
    for i in range(w_j.shape[1]):
        axarr[i+1].plot(data.index, w_j[:, i])

    axarr[i+2].plot(data.index, v)
    plt.show()

def signals_high_freq(data, w_j, zscore_thres=2.0):
    d = pd.DataFrame(w_j[:,0:5].sum(axis=1), index=data.index)
    rolling_window = d.rolling(252)
    zscore = (d - rolling_window.mean())/rolling_window.std()
    zscore = zscore.dropna()
    d = d.loc[zscore.index]
    signals_dict = {}
    curr_signal = 0
    for i in range(len(zscore)):
        score = zscore.iloc[i][0]
        sign = np.sign(score)
        if abs(score) > zscore_thres: #signal
            curr_signal = -sign
            signals_dict[zscore.index[i]] = curr_signal
        elif curr_signal != 0: #check if exit is met
            if curr_signal == 1:
                if score > 1: #exit
                    curr_signal = 0
            elif curr_signal == -1:
                if score < -1:
                    curr_signal = 0
        signals_dict[zscore.index[i]] = curr_signal

    return signals_dict

def signals_low_freq(data, w_j, n_days=60):
    d = pd.DataFrame(w_j[:,8:12].sum(axis=1), index=data.index)
    d_diff = d.diff(periods=1)
    days = 0
    prev = 0
    signals_dict = {}
    """
    for i in range(2, len(d_diff)):
        score = d_diff.iloc[i][0]
        if np.abs(score) > 1.25: #signal
            signals_dict[d_diff.index[i]] = np.sign(score)
        elif score < 0 and prev == -1:
            signals_dict[d_diff.index[i]] = -1
        elif score > 0 and prev == 1:
            signals_dict[d_diff.index[i]] = 1
        else:
            signals_dict[d_diff.index[i]] = 0
        prev = signals_dict[d_diff.index[i]]"""
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
    
def entry_exit_low_freq(signals, zscore, d):
    signal = 0
    entry_lst = []
    exit_lst = []
    for i in signals:
        s = np.sign(zscore.iloc[i])[0]
        if s > 0:
            exit = np.where(zscore.iloc[i:] <= -1.0)[0]
        else:
            exit = np.where(zscore.iloc[i:] >= 1.0)[0]
        if exit.size == 0:
            break
        exit = exit[0] + i
        entry_lst.append(d.index[i])
        exit_lst.append(d.index[exit])
    return entry_lst, exit_lst

def entry_exit(signals_df):
    signals_date = [signals_df.index[i] for i in range(1, len(signals_df)-1) if (signals_df.iloc[i-1][0] != signals_df.iloc[i][0])]
    pos_signals_date = [signals_date[i] for i in range(len(signals_date)) if signals_df.loc[signals_date[i]][0] == 1.0]
    neg_signals_date = [signals_date[i] for i in range(len(signals_date)) if signals_df.loc[signals_date[i]][0] == -1.0]
    exit_signals_date = [signals_date[i] for i in range(len(signals_date)) if signals_df.loc[signals_date[i]][0] == 0.0]
    
    return pos_signals_date, neg_signals_date, exit_signals_date

def plot_signals_high_freq(data, v, w_j):
    signals_dict = signals_high_freq(data, w_j)
    signals_df = pd.DataFrame(signals_dict, index=["Low Freq Signals"]).T
    pos_signals_date, neg_signals_date, exit_signals_date = entry_exit(signals_df)
    
    x_ord = bqplot.DateScale()
    y_sc = bqplot.LinearScale()

    line = bqplot.Lines(x=data.index, y=data.values.squeeze(), scales={'x': x_ord, 'y': y_sc},
                    stroke_width=2, display_legend=False, labels=['Underlying TS'])
    scatter1 = bqplot.Scatter(x=pd.DatetimeIndex(pos_signals_date), y=data.loc[pos_signals_date].squeeze(), colors=["green"],
                         scales={'x': x_ord, 'y': y_sc}, marker='triangle-up', default_size=25, default_opacities=[0.80])
    scatter2 = bqplot.Scatter(x=pd.DatetimeIndex(neg_signals_date), y=data.loc[neg_signals_date].squeeze(), colors=["red"],
                         scales={'x': x_ord, 'y': y_sc}, marker='triangle-down', default_size=25, default_opacities=[0.80])
    scatter3 = bqplot.Scatter(x=pd.DatetimeIndex(exit_signals_date), y=data.loc[exit_signals_date].squeeze(), colors=["white"],
                         scales={'x': x_ord, 'y': y_sc}, marker='square', default_size=25, default_opacities=[0.80])
    ax_x = bqplot.Axis(scale=x_ord)
    ax_y = bqplot.Axis(scale=y_sc, orientation='vertical', tick_format='0.2f', grid_lines='solid')

    fig = bqplot.Figure(marks=[line, scatter1, scatter2, scatter3], axes=[ax_x, ax_y])
    
    pz = bqplot.PanZoom(scales={'x': [x_ord], 'y': [y_sc]})
    pzx = bqplot.PanZoom(scales={'x': [x_ord]})
    pzy = bqplot.PanZoom(scales={'y': [y_sc], })

    #
    """  zoom_interacts = ToggleButtons(
                                            options=OrderedDict([
                                                ('xy ', pz), 
                                                ('x ', pzx), 
                                                ('y ', pzy),   
                                                (' ', None)]),
                                                icons = ["arrows", "arrows-h", "arrows-v", "stop"],
                                                tooltips = ["zoom/pan in x & y", "zoom/pan in x only", "zoom/pan in y only", "cancel zoom/pan"]
                                            )
    zoom_interacts.style.button_width = '50px'

    ResetZoomButton = Button(
        description='',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Reset zoom',
        icon='arrows-alt'
    )"""

    def resetZoom(new):
        # Reset the x and y axes on the figure
        fig.axes[0].scale.min = None
        fig.axes[1].scale.min = None
        fig.axes[0].scale.max = None
        fig.axes[1].scale.max = None  

    ResetZoomButton.on_click(resetZoom)
    ResetZoomButton.layout.width = '95%'

    link((zoom_interacts, 'value'), (fig, 'interaction'))
    display(fig, zoom_interacts)
    
def plot_signals_low_freq(data, v, w_j, n_days=60):
    signals_dict = signals_low_freq(data, w_j, n_days)
    signals_df = pd.DataFrame(signals_dict, index=["High Freq Signals"]).T
    
    pos_signals_date, neg_signals_date, exit_signals_date = entry_exit(signals_df)
    
    x_ord = bqplot.DateScale()
    y_sc = bqplot.LinearScale()

    line = bqplot.Lines(x=data.index, y=data.values.squeeze(), scales={'x': x_ord, 'y': y_sc},
                    stroke_width=2, display_legend=False, labels=['Underlying TS'])
    scatter1 = bqplot.Scatter(x=pd.DatetimeIndex(pos_signals_date), y=data.loc[pos_signals_date].squeeze(), colors=["green"],
                         scales={'x': x_ord, 'y': y_sc}, marker='triangle-up', default_size=25)
    scatter2 = bqplot.Scatter(x=pd.DatetimeIndex(neg_signals_date), y=data.loc[neg_signals_date].squeeze(), colors=["red"],
                         scales={'x': x_ord, 'y': y_sc}, marker='triangle-down', default_size=25)
    scatter3 = bqplot.Scatter(x=pd.DatetimeIndex(exit_signals_date), y=data.loc[exit_signals_date].squeeze(), colors=["white"],
                         scales={'x': x_ord, 'y': y_sc}, marker='square', default_size=25)
    ax_x = bqplot.Axis(scale=x_ord)
    ax_y = bqplot.Axis(scale=y_sc, orientation='vertical', tick_format='0.2f', grid_lines='solid')
            
    fig = bqplot.Figure(marks=[line, scatter1, scatter2, scatter3], axes=[ax_x, ax_y])
    pz = bqplot.PanZoom(scales={'x': [x_ord], 'y': [y_sc]})
    pzx = bqplot.PanZoom(scales={'x': [x_ord]})
    pzy = bqplot.PanZoom(scales={'y': [y_sc], })

    #
    """zoom_interacts = ToggleButtons(
                                            options=OrderedDict([
                                                ('xy ', pz), 
                                                ('x ', pzx), 
                                                ('y ', pzy),   
                                                (' ', None)]),
                                                icons = ["arrows", "arrows-h", "arrows-v", "stop"],
                                                tooltips = ["zoom/pan in x & y", "zoom/pan in x only", "zoom/pan in y only", "cancel zoom/pan"]
                                            )
    zoom_interacts.style.button_width = '50px'

    ResetZoomButton = Button(
        description='',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Reset zoom',
        icon='arrows-alt'
    )"""

    def resetZoom(new):
        # Reset the x and y axes on the figure
        fig.axes[0].scale.min = None
        fig.axes[1].scale.min = None
        fig.axes[0].scale.max = None
        fig.axes[1].scale.max = None  

    ResetZoomButton.on_click(resetZoom)
    ResetZoomButton.layout.width = '95%'

    link((zoom_interacts, 'value'), (fig, 'interaction'))
    display(fig, zoom_interacts)
    
def zero_padd_signals(data, signals_df):
    return {d:(signals_df.loc[d][0] if d in signals_df.index else 0.0) for d in data.index}
    
def combined_signals(data, v, w_j, n_days=60):
    lf_signals_dict = signals_low_freq(data, w_j)
    lf_signals_df = pd.DataFrame(lf_signals_dict, index=["Low Freq Signals"]).T
    lf_signals_padded = pd.DataFrame(zero_padd_signals(data, lf_signals_df), index=["High Freq Signals"]).T
    
    hf_signals_dict = signals_high_freq(data, w_j)
    hf_signals_df = pd.DataFrame(hf_signals_dict, index=["High Freq Signals"]).T
    hf_signals_padded = pd.DataFrame(zero_padd_signals(data, hf_signals_df), index=["Low Freq Signals"]).T
    
    comb_signals_df = pd.DataFrame(lf_signals_padded.values*0.5 + hf_signals_padded*0.5, index=lf_signals_padded.index)
    return comb_signals_df
   

data = pd.read_csv("C:/Users/Phili/Desktop/fond/data/OMXS30.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
data = data["Value"]
n = 1
max_levels = 0
data_len = data.shape[0]
while True:
    old_n = n
    n *= 2
    if data_len-n < 0:
        n = old_n
        break
    max_levels += 1
data = data.iloc[-n:]

v, w_j = mra(data, 'db4', max_levels)
#plot_signals_low_freq(data, v, w_j)
#plot_signals_high_freq(data, v, w_j)
print(v)
print(w_j)
plot_mra(data, v , w_j)