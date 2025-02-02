import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA as ta
import yfinance as yf
from pandas._config import display
import statistics as stat
import Backtesting_Methods as meth
import math
from matplotlib.gridspec import GridSpec

## Tickers
#Gold Future: GC=F
#Soybeans Future: ZS=F
#GBPJPY
#Natural Gas Future: NG=F





######      Scenario specifications

#Simple Moving Average (SMA)
sma_length = 14
sma4_length = 8

#Relative Strengh Index (RSI)
rsi_length = 7
rsi_sma_length = 14
rsi_smooth_length = 7       # "Smooth" is here just a second SMA on the RSI

#Stochastic RSI
stochrsi_length = rsi_length
stochrsi_sma_length = rsi_sma_length

#Bollinger Bands
bollinger_length = 4

#Threshold for "integration" filters
diff_threshold = 1
diff_threshold_tupel = (0.01, 0.01)

# Asset and time frame
symbol = "GOOG"
start_date = "2024-01-01"
end_date = "2024-12-31"




######      Definition of a function that generates signals and backtests strategy based on passed specifications
def szenario(
rsi_len = rsi_length,
rsi_sma_len = rsi_sma_length,
rsi_smooth_len = rsi_smooth_length,
stochrsi_len = stochrsi_length,
stochrsi_sma_len = stochrsi_sma_length,
sma_len = sma_length,
sma4_len = sma4_length,
bollinger_len = bollinger_length,
diff_thres_tu = diff_threshold_tupel,
diff_thres = diff_threshold,
sym = symbol,
start = start_date,
end = end_date):


    # Data download via yFinance
    data = pd.DataFrame(yf.download(sym, start=start, end=end))
    data.columns = [col[0].lower() for col in data.columns]

    
    ###     Adding technical indicators to the data (un-comment those who needed)

    #Simple moving averages
    data["sma"] = ta.SMA(data, sma_len, "close")
    #data["sma_close"] = ta.SMA(data, sma4_len, "close")
    #data["sma_open"] = ta.SMA(data, sma4_len, "open")
    #data["sma_high"] = ta.SMA(data, sma4_len, "high")
    #data["sma_low"] = ta.SMA(data, sma4_len, "low")

    # Relative strengh index (RSI)
    rsi = ta.RSI(data, rsi_len, "close")
    data['rsi'] = rsi

    # Different SMAs of the RSI
    data["rsi_sma"] = ta.SMA(data, rsi_sma_len, "rsi")
    data["rsi_smooth"] = ta.SMA(data, rsi_smooth_len, "rsi")

    # Average True Range (ATR) -> measures volatility
    #data["ATR"] = ta.ATR(data, 14)
    #data["ATR_sma"] = ta.SMA(data, 14, column="ATR")

    # Stochastic RSI
    data["stochrsi"] = ta.STOCHRSI(data, 14, 14)
    data["stochrsi_sma"] = ta.SMA(data, stochrsi_sma_length, "stochrsi")
    data = data.fillna({"stochrsi": 0})

    # Open-to-Low an open-to-high ratios on daily basis
    #data["%opentoLow"] = [x / y for (x, y) in zip(data["Low"], data["Open"])]
    #data["%opentoHigh"] = [x / y for (x, y) in zip(data["High"], data["Open"])]

    # Bollinger Bands
    #bollinger = ta.BBANDS(data, bollinger_len, column="close")
    #data = data.join(bollinger)


    # Replace NaN values with 0
    data.fillna(0, inplace=True)


    m = max(rsi_len, rsi_sma_len, rsi_smooth_len)     # Earlierst plottet data point -> needed because early datapoints are depending on indicator not usable yet (here RSI-length)
    M = len(data)    # Latest used data points

    u = 50          # Upper minimum bound for RSI
    l = 50          # Lower maximum bound for RSI



    #######     Generation of signals -> Method to use to use for signals has to be un-commented

    #Basic RSI-cross based signals with restriction of upper/lower RSI-bounds
    #data["Signal"] = meth.rsi_crosses_cap(data["rsi_smooth"][:], data["rsi_sma"][:], 0, 100)

    #Basic RSI-cross based signals without restirction
    data["Signal"] = meth.rsi_crosses_nocap(data["rsi_smooth"][:], data["rsi_sma"][:])
    
    #Basic RSI based signal with upper/lower bounds with own Difference-Integration-filter
    #data["Signal"] = meth.rsi_diff_integration(data, diff_thres, u, l)

    #Basic RSI-cross based signals without restriction
    #data["Signal"] = meth.stochrsi_crosses_nocap(data["stochrsi"], data["stochrsi_sma"])

    #Basic RSI based signal without upper/lower bounds with own Difference-Integration-filter
    #data["Signal"], stochrsi_avg, stochrsi_sum= meth.stochrsi_diff_integration(data, diff_thres_tu)

    #Bollinger based signals -> not yet implemented
    #data["Signal"] = meth.NAME(data["rsi_smooth"][:], data["rsi_sma"][:])


    # Reindexing data of proper length
    data = data.iloc[m:M]
    data["index"] = list(range(0, len(data)))
    data.set_index("index", inplace=True)



    ######      Backtesting the strategy
    backtest = meth.backtest_moneyinput(data, money=100, split=1, loss_limit_per=0, max_ticks=0)
    #backtest = meth.backtest_1persignal(data, max=10000)
    #backtest = meth.backtest_allinever(data, money=100)


    ######      Call the visualization of the scenario
    meth.scenario_visu(data, backtest, m, M, rsi_len = rsi_len,
        rsi_sma_len = rsi_sma_len,
        rsi_smooth_len = rsi_smooth_len,
        sma_len = sma_len,
        bollinger_len = bollinger_len,
        diff_thres = diff_thres,
        sym = symbol,
        start = start_date,
        end = end_date)

    return backtest[0], backtest[1]




######      Single Test

scen = szenario(
rsi_len = rsi_length,
rsi_sma_len = rsi_sma_length,
rsi_smooth_len = rsi_smooth_length,
sma_len = sma_length,
bollinger_len = bollinger_length,
diff_thres = diff_threshold,
sym = symbol,
start = start_date,
end = end_date)

print("Realized profits:\t\t\t\t\t\t\t\t", scen[0])
print("Realized profits + balance of open positions:\t", scen[0]+scen[1])


