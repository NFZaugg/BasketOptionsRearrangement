import numpy as np
from helper import mcOption, implyVol
from hagan import calibrationHagan
import itertools
import matplotlib.pyplot as plt
from helper import (
    F_inverse_S,
    OptionPrices,
    blackScholes,
    impVol,
    shuffleSamples,
    concatSamples,
)
from matplotlib.axes._axes import Axes as MplAxes
from importHelper import (
    getIVTableForMaturity,
    readSpotPrices,
    getIndexIVForMaturity,
    readBidAsk,
)
from scipy.interpolate import interp1d

# Parameters
nSamples = 20000
nBins = 1600
nRuns = 5
r = 0
isSecond = False  # If isSecond, the algorithm runs on 1.5y,2y, else on 3m,6m,1y

# Constants
DJI_DIV = 0.151727526
W = 1 / (100 * DJI_DIV)
S = readSpotPrices()
INDEX_SPOT = S.sum() * W

# Preliminary
marker = itertools.cycle(("x", "+", ".", "o", "*"))
plotSpine = np.linspace(0.8, 1.2, 100)


# PlotFunctions
def PlotMarketVsModelIV(
    plotSpine: np.ndarray,
    T: float,
    basketOptionGrid: np.ndarray,
    basketOptionPrice: np.ndarray,
    ax: MplAxes,
    label: str,
    paramsSABRIndex: np.ndarray,
    paramsSABRBid: np.ndarray,
    paramsSABRAsk: np.ndarray,
) -> None:
    vol, success = implyVol(basketOptionPrice, INDEX_SPOT, basketOptionGrid, 0.1, T, r)
    mark = range(0, 100, 10)
    if success:
        ax.plot(
            plotSpine,
            vol * 100,
            label="Model Skew",
            markevery=mark,
            marker="s",
            linewidth=2,
            color="#00539C",
        )
        ax.plot(
            plotSpine,
            impVol(basketOptionGrid, INDEX_SPOT, T, paramsSABRIndex) * 100,
            label="Market Skew - Mid",
            markevery=mark,
            marker="o",
            linewidth=2,
            color="orange",
        )
        ax.plot(
            plotSpine,
            impVol(basketOptionGrid, INDEX_SPOT, T, paramsSABRBid) * 100,
            label="Market Skew - Ask/Bid",
            color="black",
            linestyle="dashed",
        )
        ax.plot(
            plotSpine,
            impVol(basketOptionGrid, INDEX_SPOT, T, paramsSABRAsk) * 100,
            color="black",
            linestyle="dashed",
        )
        # ax.plot(ask.index, ask.values, color="black", linestyle='dashed')
        ax.set_xlabel("Strike (relative to ATM) ")
        ax.set_ylabel("Implied Volatility [%]")
        ax.set_xlim([0.8, 1.2])
        ax.grid(True)
        ax.set_title(label)
        l = ax.legend(prop={"size": 10})
        l.get_frame().set_edgecolor("black")
    else:
        print("Failed Optimizing")


# Target Creation
def getIndependentSamples(maturity: float, optionValue: OptionPrices) -> np.ndarray:
    x = np.random.uniform(size=(nSamples, len(S)))
    y = F_inverse_S(maturity, x, optionValue).T
    return y * W


def getTargetDensity(maturity: float, paramsSABRIndex: np.ndarray) -> np.ndarray:
    spineFromParams = np.linspace(0.01, 3, nBins)
    spine = np.linspace(0, 3, nBins)
    gridForOption = np.atleast_2d(spineFromParams).T * np.atleast_2d(INDEX_SPOT)
    optionValue = blackScholes(
        INDEX_SPOT,
        gridForOption,
        maturity,
        0,
        impVol(gridForOption, INDEX_SPOT, maturity, paramsSABRIndex),
    )
    grid = spine * INDEX_SPOT
    optionValueInterp = interp1d(
        gridForOption.squeeze(),
        optionValue.squeeze(),
        bounds_error=False,
        fill_value="extrapolate",
    )(grid)
    return grid, (np.diff(optionValueInterp, axis=0) / np.diff(grid, axis=0) + 1)


def runAlgorithm(
    bins: np.ndarray,
    samples: np.ndarray,
    targetPerBin: np.ndarray,
    stashedSamples: np.ndarray,
) -> np.ndarray:
    for j in range(nRuns):
        samples = np.sort(samples, axis=0)
        samples, stashedSamples = selectSamplesInBins(bins, samples, targetPerBin, stashedSamples)
        shuffleSamples(samples)
        samples, stashedSamples = selectSamplesInBins(bins, samples, targetPerBin, stashedSamples)
    stashedSamples = stashedSamples[1:]
    return concatSamples(stashedSamples, samples).sum(axis=1)


def selectSamplesInBins(
    bins: np.ndarray,
    samples: np.ndarray,
    targetPerBin: np.ndarray,
    stashedSamples: np.ndarray,
) -> tuple[np.ndarray]:
    for i in range(len(bins) - 1):
        basketPrices = samples.sum(axis=1)
        if targetPerBin[i] > 0:
            # Select samples in bin
            mask = np.ones(len(samples), bool)
            valid = np.where((basketPrices > bins[i]) & (basketPrices <= bins[i + 1]))[0]
            if len(valid) != 0:
                mask[np.random.choice(valid, size=len(valid), replace=False)[: targetPerBin[i]]] = 0
            stashedSamples = concatSamples(stashedSamples, samples[~mask])
            targetPerBin[i] -= sum(~mask)
            samples = samples[mask]
    return samples, stashedSamples


def main() -> None:
    bids, asks = readBidAsk()
    name = "Graphs/1st.png" if not isSecond else "Graphs/2nd.png"
    _, axes = plt.subplots(ncols=2 if isSecond else 3, figsize=(12, 6))
    labels = ["3m", "6m", "1y", "1.5y", "2y"]
    mats = [3 / 12, 6 / 12, 1] if not isSecond else [1.5, 2]
    for pos, maturity in enumerate(mats):
        ax = axes[pos]
        bid = bids.iloc[3 * isSecond + pos]
        ask = asks.iloc[3 * isSecond + pos]
        label = labels[3 * isSecond + pos]
        INDEX_IV = getIndexIVForMaturity(maturity)
        IV_TABLE_ACTUAL = getIVTableForMaturity(maturity, "data/vol_members.csv")
        spine, ivs = np.array(IV_TABLE_ACTUAL[0]), np.array(IV_TABLE_ACTUAL[1:]) / 100
        # Calibrate SABR Params
        paramsSABRConsituents = np.array(
            [calibrationHagan(S[i] * spine, maturity, 0.9, ivs[i], S[i], ivs[i][5]) for i in range(len(ivs))]
        )

        paramsSABRIndexMid = np.atleast_2d(
            calibrationHagan(
                INDEX_SPOT * spine,
                maturity,
                0.9,
                INDEX_IV.values / 100,
                INDEX_SPOT,
                INDEX_IV.values[5] / 100,
            )
        )
        paramsSABRIndexBid = np.atleast_2d(
            calibrationHagan(
                INDEX_SPOT * spine,
                maturity,
                0.9,
                bid.values / 100,
                INDEX_SPOT,
                bid.values[5] / 100,
            )
        )
        paramsSABRIndexAsk = np.atleast_2d(
            calibrationHagan(
                INDEX_SPOT * spine,
                maturity,
                0.9,
                ask.values / 100,
                INDEX_SPOT,
                ask.values[5] / 100,
            )
        )

        # Generate initial samples from marginal distributions
        optionValue = OptionPrices(noOfGridPoints=10000, spotPrices=S, sabrParams=paramsSABRConsituents)
        samples = getIndependentSamples(maturity, optionValue)

        # Generate target vector
        bins, target = getTargetDensity(maturity, paramsSABRIndexMid)
        basketSamplesFromDist = interp1d(target, bins[1:], bounds_error=False, fill_value=0)(
            np.linspace(0, 1, nSamples)
        )
        targetVector, _ = np.histogram(basketSamplesFromDist, bins=bins)

        # Match first moment of constituents and index
        samples = samples + (
            basketSamplesFromDist[basketSamplesFromDist != 0].mean() - samples.sum(axis=1).mean()
        ) / len(S)

        # Run the main algorithm
        basketSamples = runAlgorithm(bins, samples, targetVector, np.zeros((1, 30)))

        # Evaluate options
        basketOptionGrid = plotSpine * S.sum() * W
        modelVanillaOptionPrices = mcOption(basketSamples, basketOptionGrid, maturity, r)

        # Plot
        PlotMarketVsModelIV(
            plotSpine,
            maturity,
            basketOptionGrid,
            modelVanillaOptionPrices,
            ax,
            label,
            paramsSABRIndexMid,
            paramsSABRIndexBid,
            paramsSABRIndexAsk,
        )

    plt.tight_layout()
    plt.savefig(name, bbox_inches="tight", dpi=400)
    plt.show()


if __name__ == "__main__":
    main()
