import pandas as pd
import numpy as np

COLS = [0.8, 0.85, 0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1, 1.15, 1.2]
ROWS = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 1.5, 2]
NOOFCONS = 30


def readSpotPrices():
    return pd.read_csv("data/spotPrices.csv").values.flatten()


def readCSV(fileName, rows, cols, noOfConstituents):
    data = pd.read_csv(fileName)
    data.index = rows * noOfConstituents
    data.columns = cols
    return [data.iloc[i * 7 : i * 7 + 7] for i in range(noOfConstituents)]


def getIVTableForMaturity(maturity: float, fileName):
    frames = readCSV(fileName, ROWS, COLS, NOOFCONS)
    header = [list(frames[0].columns)]
    table = [list(frame.loc[maturity].values) for frame in frames]
    return header + table


def getIVTable(fileName):
    frames = readCSV(fileName, np.linspace(1, 5, 5), COLS, 1)
    header = list(frames[0].columns)
    table = frames[0]
    return [header, table]


def readBidAsk():
    frame = pd.read_csv("data/bidask.csv")
    frame.columns = COLS
    ask = frame.iloc[0:5]
    bid = frame.iloc[5:]
    return bid, ask


def getIndexIVForMaturity(maturity: float):
    frame = pd.read_csv("data/vol_index.csv")
    frame.index = ROWS
    frame.columns = COLS
    return frame.loc[maturity]


if __name__ == "__main__":
    data = readBidAsk()
