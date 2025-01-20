# TradingSignalGenerationAndBacktesting

## Project scope and motivation
When being interested in trading, at some point you get in touch with technical indicators. Although pure technical analysis lacks quality for reasoning investment decisions, they can be helpful for timing entry and exit of positions. Earlier this year I executed a short trade on Daimler Truck after reviewing qualitative aspects, like the overall market situation, and timed the entry based on relative strength indicators. I Closed the position half a year later with profits of 22%. My idea was to implement a library for generating signals from technical data and back testing the resulting strategies. Of course, also non-technical signals can be generated, when logic and suitable data are provided.

## Implemented aspects
- Finding and transforming the right data
- Variability in indicators, asset symbols, back testing approaches, signal-generation and -filtering
- Visualization of used data, indicators, entries/exists and the corresponding equity curve

## Ideas for further expansions
- Add more pre-defined logic bricks for signal generation
- Usage of broader data (e.g. macro data or company specifics in case of equities)
- Automating back testing and optimization of parameters (e.g. walk-forward analysis through historical data)
- Differentiation of generated and traded signals in the plots

## Example using daily closing prices of gold futures
Via yFinance daily data on prices of gold futures (symbol “GC=F”) is downloaded for the entire year 2024. The back test simulates being constantly in the market with one short or one long position of 100 money units without reinvesting realized profits. So, if one long position is closed, automatically a new short position is opened and vice versa. If on one signal follows another one of the same type (e.g. long and long), only the first one is executed and the position remains open until an opposite signal occurs. The script calculates the seven-day relative strength (RSI7) of the asset price and the three-day simple moving average (SMA3) of this RSI7. The fundamental logic is to indicate a long entry if is SMA3 crosses the RSI7 from below and the RSI7. Entering a short position corresponds oppositely to such a crossing from above and a RSI7.
This example represents the standard-setting of the code provided in my GitHub repository. Detailed information on single trades simulated is printed in the console when executing the “scenario_sim.py”.

## Comment on code usage
"scenario_sim.py" contains the core, "Backtesting_Methods.py" contains neccessary functions of different purposes written by me.



