import statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA as ta
import yfinance as yf
import math
from matplotlib.gridspec import GridSpec



# Generating Relative Strength signals with a restriction -> long entries require a maximum RSI and short entries a minimum
def rsi_crosses_cap(rsi, sma, u, l):
    #print(rsi)
    ret = len(rsi)*[0]

    for i in range(len(rsi)-1):

        if rsi[i] >= sma[i] and rsi[i + 1] <= sma[i + 1] and rsi[i+1] > u:
            ret[i + 1] = -1
        elif rsi[i] <= sma[i] and rsi[i + 1] >= sma[i + 1] and rsi[i+1] < l:
            ret[i + 1] = 1
        else:
            ret[i + 1] = 0

    return np.array(ret)



# Generating Relative Strength signals WITHOUT a restriction on the RSI
def rsi_crosses_nocap(rsi, sma):
    #print(rsi)
    ret = len(rsi)*[0]


    for i in range(len(rsi)-1):

        if rsi[i] >= sma[i] and rsi[i+1] <= sma[i+1]:
            ret[i+1] = -1
        elif rsi[i] <= sma[i] and rsi[i+1] >= sma[i+1]:
            ret[i+1] = 1
        else:
            ret[i+1] = 0

    return np.array(ret)



# Generated Stochastic-Relative Strength signals WITHOUT a restriction on the RSI
def stochrsi_crosses_nocap(stochrsi, stochrsi_sma):
    #print(rsi)
    ret = [0]


    for i in range(len(stochrsi)-1):

        if stochrsi[i] <= stochrsi_sma[i] and stochrsi[i+1] >= stochrsi_sma[i+1]:
            ret.append(1)
            last_trade_rsi = stochrsi[i+1]
        elif stochrsi[i] >= stochrsi_sma[i] and stochrsi[i+1] <= stochrsi_sma[i+1]:
            ret.append(-1)
            last_trade_rsi = stochrsi[i+1]
        else:
            ret.append(0)

    return np.array(ret)


###### not yet finished Bollinger-Band signals
def rsi_crosses_bb(data):
    #print(rsi)
    ret = []

    sma = data["sma"]
    rsi = data["rsi"]
    bb_u = data["BB_UPPER"]
    bb_m = data["BB_MIDDLE"]
    bb_l = data["BB_LOWER"]

    for i in range(len(rsi)-1):
        if rsi[i] <= sma[i] and rsi[i+1] >= sma[i+1]:
            ret.append(1)
        elif rsi[i] >= sma[i] and rsi[i+1] <= sma[i+1]:
            ret.append(-1)
        else:
            ret.append(0)

    ret.append(0)

    return np.array(ret)


# Signal generator based on "rsi_crosses_nocap", BUT: Calculates the sum of differences between the RSI and its SMA
# between each two OPPOSITE signals -> intuition is something like "integrating" the surface between the lines, between
# the crossing points. Signals with such a sum below a threshold count as not valid enough and get cancelled.
def rsi_diff_integration(data, diff_thres, u, l):

    rsi = data["rsi"]
    rsi_sma = data["rsi_sma"]
    diff = [abs(x-y) for x, y in zip(rsi, rsi_sma)]

    rsi_action = rsi_crosses_cap(rsi, rsi_sma, u, l)
    action_diff = rsi_action.copy()
    new_action = rsi_action.copy()

    for i in range(len(rsi_action)):

        cnt = 0

        if abs(rsi_action[i]) == 1 and cnt == 0:

            for j in range(i+1, len(rsi_action)):

                if abs(rsi_action[j]) == 1 and cnt == 0:

                    cnt = 1
                    S = 0

                    for k in range(i, j):

                        for l in diff[i:j]:
                            S += abs(l)
                        action_diff[j] = (-1*rsi_action[j]*S)/(j-i) # avg of sum Method
                        #action_diff[j] = (rsi_action[j] * S) # just sum Method

    for i in range(len(new_action)):
        if abs(action_diff[i]) < diff_thres:
            new_action[i] = 0

    return new_action


# As above, based on stochastic RSI ("stochrsi_crosses_nocap")
def stochrsi_diff_integration(data, diff_thres):

    rsi = data["stochrsi"]
    rsi_sma = data["stochrsi_sma"]
    diff = [x-y for x, y in zip(rsi, rsi_sma)]

    rsi_action = rsi_crosses_nocap(rsi, rsi_sma)
    action_diff_avg = [float(rsi_action[i]) for i in range(len(rsi_action))]
    action_diff_sum = [float(rsi_action[i]) for i in range(len(rsi_action))]
    new_action = rsi_action.copy()

    first_marker = -1 #Marker um erstes Signal wegen mangel an Infos ruszunehmen

    for i in range(len(rsi_action)):

        cnt = 0


        if abs(rsi_action[i]) == 1 and cnt == 0:

            if first_marker == -1: first_marker = i

            for j in range(i+1, len(rsi_action)):

                if abs(rsi_action[j]) == 1 and cnt == 0:

                    S = 0

                    for l in diff[i:j]:
                        #print(l)
                        S += l

                    action_diff_avg[j] = (S)/(j-i) # avg Method
                    action_diff_sum[j] = (S) # Sum Method

    for i in range(len(new_action)):
        if abs(action_diff_avg[i]) < diff_thres[0] or abs(action_diff_sum[i]) < diff_thres[1]:
            new_action[i] = 0

    action_diff_avg[first_marker] = 0
    action_diff_sum[first_marker] = 0
    new_action[first_marker] = 0

    return new_action, action_diff_avg, action_diff_sum







######### BACKTESTING FUNCTIONS


# I also experimented with other allocation methods, but this frist one was the most straight forward.
# Inputs are a starting value of money, split = number of maximum positions on which the money is equally spread via
# signals following another,percentual maximum-loss restriction and a maximum of ticks a position will be open
def backtest_moneyinput(data, money=100, split=1, loss_limit_per=0, max_ticks=0):

    # loss_limit_per=0 means no restriction of losses
    # max_ticks=0 means theoretically forever opened positions

    long = []
    short = []
    balance = 0
    end_position_balance = 0
    m = money
    equity_curve = []
    returns_per_trade = []

    print("[Day] [Open LONG Positions] [Open SHORT Positions] [Current Profits] [Current Asset Price]")

    for i in range(len(data["Signal"])):

        A = data["Signal"][i]
        p = data["close"][i]


        if A == 1:

            for j in range(len(short)):
                change = ((short[j]-p)/short[j])*(money/split)
                balance += change
                returns_per_trade.append(change)


            m += len(short) * (money / split)
            short = []

            if len(short) + len(long) < split:
                long.append(p)
                m -= money/split


            print("[",i,"]", long, short, balance, p)

        elif A == -1:

            for j in range(len(long)):
                change = (((p-long[j])/long[j])*(money/split))
                balance += change
                returns_per_trade.append(change)

            m += len(long) * money / split
            long = []

            if len(short) + len(long) < split:
                short.append(p)
                m -= money / split

            print("[",i,"]", long, short, balance, p)

        elif loss_limit_per!=0: ###   Loss-Limit

            for j in range(len(long)):
                change = ((p-long[j])/long[j])*(money/split)
                if change <= -loss_limit_per*money/split:
                    balance += change
                    returns_per_trade.append(change)
                    del long[j]

            for j in range(len(short)):
                change = ((short[j]-p)/short[j])*(money/split)
                if change <= -loss_limit_per*money/split:
                    balance += change
                    returns_per_trade.append(change)
                    del short[j]

        elif max_ticks != 0 and A == 0 and split == 1: ###   Maximum an Ticks einer Position, nur bei einer Position gleichzeitig

            cnt = 1

            for k in range(i, 0, -1):

                if data["Signal"][k] == 0:
                    cnt += 1
                    continue
                elif abs(data["Signal"][k]) == 1 and cnt >= max_ticks:

                    if len(long) > 0:
                        change = ((p-long[0])/long[0])*(money/split)
                        balance += change
                        returns_per_trade.append(change)
                        del long[0]
                    elif len(short) > 0:
                        change = ((p - short[0]) / short[0]) * (money / split)
                        balance += change
                        returns_per_trade.append(change)
                        del short[0]
                    k = 0


        equity_curve.append(money+balance)


    for j in range(len(short)):
        end_position_balance += ((short[j]-p)/short[j])*(money/split)

    for j in range(len(long)):
        end_position_balance += ((p-long[j])/long[j])*(money/split)




    return balance, end_position_balance, equity_curve, returns_per_trade


# This backtesting function opens a position of one contract per share  on every signal, as long as the sum of money
# invested is lower than the passed maximum value "max"
def backtest_1persignal(data, max):

    long = []
    short = []
    balance = 0
    end_position_balance = 0

    print("[Day] [Open LONG Positions] [Open SHORT Positions] [Current Profits] [Current Asset Price]")

    if max < math.inf:
        equity_curve = [max]
    else:
        equity_curve = [0]

    for i in range(len(data["Signal"])):

        A = data["Signal"][i]
        p = data["close"][i]



        if A == 1:

            for j in range(len(short)):
                balance += short[j]-p
            short = []

            if sum(long) <= max:
                long.append(p)

            print("[",i,"]", long, short, balance, p)

        elif A == -1:

            for j in range(len(long)):
                balance += p-long[j]
            long = []

            #if sum(short) <= max:
                #short.append(p)

            print("[",i,"]", long, short, balance, p)

        equity_curve.append(equity_curve[0]+balance)


    for j in range(len(short)):
        end_position_balance += short[j]-p

    for j in range(len(long)):
        end_position_balance += p-long[j]

    del equity_curve[0]

    return balance, end_position_balance, equity_curve


# "backtest_allinever" works similar to "backtest_moneyinput", but on every position opened
# all of the available money is invested
def backtest_allinever(data, money):

    long = []
    short = []
    balance = money
    end_position_balance = 0
    current_trade = 0
    cnt_trades = 0
    equity_curve = []
    print("[Day] [Open LONG Positions] [Open SHORT Positions] [Current Profits] [Current Asset Price]")

    for i in range(len(data["Signal"])):

        A = data["Signal"][i]
        p = data["close"][i]


        if A == 1:
            cnt_trades += 1
            for j in range(len(short)):
                balance = (1+abs(1-(p/short[j])))*current_trade
            short = []

            if len(long)+len(short) < 1:
                long.append(p)
                current_trade = balance
                balance = 0

            print("[",i,"]", long, short, current_trade-money, p)

        elif A == -1:
            cnt_trades += 1
            for j in range(len(long)):
                balance = ((current_trade)/long[j])*p
            long = []

            if len(long)+len(short) < 1:
                short.append(p)
                current_trade = balance
                balance = 0

            print("[",i,"]", long, short, current_trade-money, p)

        if cnt_trades == 0:
            equity_curve.append(money)
        else:
            equity_curve.append(current_trade)


    balance = current_trade

    for j in range(len(short)):
        end_position_balance = (1+abs(1-(p/short[j])))*current_trade-balance

    for j in range(len(long)):
        end_position_balance = (current_trade/long[j])*p-balance


    return balance-money, end_position_balance, equity_curve




######### HELP FUNCTIONS
normalize = lambda l, f : [i/float(f) for i in l]

array_trafo_dim1zuDIM1 = lambda l : [i[0] for i in l]

def hochk_neg(x,k):
    if x>=0: return pow(x,k)
    else: return pow(x,k-1)*(-x)




######### VISUALIZATION
## This function has to be redesigned, so that you dont have to do manual adjustments of the visualization function in case of 
## changing the calculation of indicators

def scenario_visu(data, backtest, m, M, rsi_len,
rsi_sma_len,
rsi_smooth_len,
sma_len,
bollinger_len,
diff_thres,
sym,
start,
end):

    ###     Plot Definition
    fig = plt.figure(figsize=(14, 18))
    gs = GridSpec(3, 1, height_ratios=[2, 2, 1])

    # Subplots definieren
    ax1 = fig.add_subplot(gs[0, 0])  # Upper Plot
    ax2 = fig.add_subplot(gs[1, 0])  # Middle Plot
    ax3 = fig.add_subplot(gs[2, 0])  # Lower Plot

    ax1.plot(data['close'])
    # axs[0].plot(data['Open'][m:M])
    # axs[0].plot(data['High'][m:M])
    # axs[0].plot(data['Low'][m:M])
    #ax1.plot(data["sma"])
    # axs[0].plot(data["sma_close"][m:M])
    # axs[0].plot(data["sma_open"][m:M])
    # axs[0].plot(data["sma_high"][m:M])
    # axs[0].plot(data["sma_low"][m:M])
    # axs[0].plot(data["BB_UPPER"][m:M])
    # axs[0].plot(data["BB_MIDDLE"][m:M])
    # axs[0].plot(data["BB_LOWER"][m:M])
    long = np.where(data["Signal"] == 1, data["close"], None)
    short = np.where(data["Signal"] == -1, data["close"], None)

    ax1.scatter(list(range(0, M - m)), long, color="green", zorder=3)
    ax1.scatter(list(range(0, M - m)), short, color="red", zorder=4)

    #ax1.legend(["Closing Price", str("SMA(Close, " + str(sma_len) + ")")], bbox_to_anchor=(1, 1), fontsize=20)
    ax1.legend(["Closing Price", "Long Signal", "Short Signal"], bbox_to_anchor=(1, 1), fontsize=20)
    ax1.tick_params(labelsize=20)

    #ax2.plot(data["rsi"])
    ax2.plot(data["rsi_sma"], zorder=1)
    ax2.plot(data["rsi_smooth"], zorder=2)
    # axs[1].plot(data["stochrsi"][m:M])
    # axs[1].plot(data["stochrsi_sma"][m:M])
    # axs[2].plot(data["sma4_diff_norm_sma"][m:M])
    # axs[1].plot(data["ATR"][m:M])
    # axs[1].plot(data["ATR_sma"][m:M])

    long = np.where(data["Signal"] == 1, data["rsi_sma"], None)
    short = np.where(data["Signal"] == -1, data["rsi_sma"], None)

    ax2.scatter(list(range(0, M - m)), long, color="green", zorder=3)
    ax2.scatter(list(range(0, M - m)), short, color="red", zorder=4)

    ax2.legend(["SMA(RSI," + str(rsi_sma_len)+")", "SMA(RSI, "+ str(rsi_smooth_len)+ ")", "Long Signal", "Short Signal"], bbox_to_anchor=(1.35, 1),
               fontsize=20)
    #ax2.hlines(y=50, xmin=0, xmax=M, color="black")
    ax2.tick_params(labelsize=20)


    # Simple plotting of signals -> 0 = nothing, 1 = go long, -1 = go short
    # axs[2].scatter(list(range(m,M)), data["Signal"][m:M], color="red")


    # axs[1].axhline(y=stat.median(data["sma4_diff_norm"])*1.12, color='b', linestyle='-')
    # axs[1].axhline(y=u, color='b', linestyle='-')
    # axs[1].axhline(y=l, color='b', linestyle='-')
    # axs[1].legend(["rsi_sma", "rsi_smooth"])

    # axs[2].plot(stochrsi_avg[m:M])
    # axs[3].plot(stochrsi_sum[m:M], color="b")

    # axs[2].plot(data["sma4_diff_derivative"][m:M])
    # axs[3].plot(data["sma4_diff_derivative_sma"][m:M])
    # axs[3].axhline(y=-0.7, color='b', linestyle='-')
    # axs[3].axhline(y=0.7, color='b', linestyle='-')


    ax3.plot(backtest[2])
    ax3.legend(["Equity Curve"], bbox_to_anchor=(1, 1), fontsize=20)
    ax3.tick_params(labelsize=20)


    plt.tight_layout()
    plt.show()

