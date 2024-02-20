import numpy as np
from scipy.optimize import root
from scipy.stats import norm
from hagan import haganImpliedVolatility
from scipy.interpolate import interp1d

N = norm.cdf

class OptionPrices:
    """Stores Option prices from sabr params"""
    def __init__(self, noOfGridPoints, spotPrices, sabrParams):
        self.noOfGridPoints = noOfGridPoints
        self.stored = {}
        self.spotPrices = spotPrices
        self.sabrParams = sabrParams

    def getGridAndValues(self, T: float):
        if T in self.stored.keys():
            return self.stored[T]
        else:
            spine = np.linspace(0.00001, 3, self.noOfGridPoints)
            grid = np.atleast_2d(spine).T * np.atleast_2d(self.spotPrices)
            optionValue = blackScholes(
                self.spotPrices,
                grid,
                T,
                0,
                impVol(grid, self.spotPrices, T, self.sabrParams),
            )
            self.stored[T] = (grid, optionValue)
            return self.stored[T]

def concatSamples(samples: np.ndarray, samplesToAdd: np.ndarray) -> np.ndarray:
    return np.concatenate([samples, samplesToAdd])


def shuffleSamples(samples: np.ndarray)-> np.ndarray:
    for col_index in range(samples.shape[1]):
        np.random.shuffle(samples[:, col_index])


def mcOption(samples: np.ndarray, K: np.ndarray, T: float, r: float, isCall: bool = True) -> np.ndarray:
    cpSign = 1 if isCall else -1
    payoff = np.maximum(cpSign * (samples[:, np.newaxis] - K), 0)
    return (payoff * np.exp(-r * T)).mean(axis=0)


def implyVol(V: np.ndarray, S_0: float, K: np.ndarray, sigma: np.ndarray, tau: float, r: float, isCall=True) -> np.ndarray:
    sigmaInit = np.full(len(K), sigma)
    optPrice = lambda sigma: blackScholes(S_0, K, tau, r, sigma, isCall)
    optimized = [root(lambda sigma: V[j] - optPrice(sigma)[j], sigmaInit[j]) for j in range(len(K))]
    success = np.array([optim.success for optim in optimized]).all()
    return np.array([optim.x for optim in optimized]), success


def blackScholes(
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    sigma: np.ndarray,
    isCall: bool = True,
) -> np.ndarray:
    """Calculates BS price for array of strikes and vols"""
    if isinstance(K, np.ndarray):
        if len(K.shape) == 2:
            S = np.repeat(np.atleast_2d(S), K.shape[0], axis=0)
        else:
            S = np.repeat(S, K.shape[0], axis=0)
    pc = 1 if isCall else -1
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(pc * d1) - K * np.exp(-r * T) * N(pc * d2)


def pdfOfStocks(T: float, optionData : OptionPrices) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the CDF of the individual stocks"""
    grid, value = optionData.getGridAndValues(T)
    return grid[1:-1], np.diff(np.diff(value, axis=0) / np.diff(grid, axis=0), axis=0) / np.diff(grid[:-1], axis=0)




def impVol(K: np.ndarray, S: np.ndarray, T: float, sabr: np.ndarray) -> np.ndarray:
    length = 1 if isinstance(S, float) else len(S)
    noOfStrikes = 1 if isinstance(K, float) else K.shape[0]

    return np.array(
        [haganImpliedVolatility(np.atleast_2d(K.T)[:, i], T, S, *sabr[:length].T) for i in range(noOfStrikes)]
    )


def cdfOfStocks(T: float, optionData: OptionPrices) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the CDF of the individual stocks"""
    grid, value = optionData.getGridAndValues(T)
    return grid[:-1], np.diff(value, axis=0) / np.diff(grid, axis=0) + 1


def F_inverse_S(T: float, x: np.ndarray, optionData: OptionPrices) -> np.ndarray:
    """
    Inverse of CDF of individual stocks
    """
    grid, values = cdfOfStocks(T, optionData)
    # for i in range(len(values[0])):
    #     values[:, i], index = np.unique(values[:, i], return_index=True)
    #     grid[:, i] = grid[index, i]
    # return np.array([UnivariateSpline(values[:, i], np.log(grid[:, i]))(x) for i in range(len(values[0]))])
    interpolators = []
    for i in range(len(values[0])):
        # Filter out non-increasing values
        increasing_mask = np.diff(values[:, i]) > 0
        filtered_x = values[:, i][:-1][increasing_mask]
        filtered_y = grid[:, i][:-1][increasing_mask]
        seen = set()
        filtered_x, filtered_y = zip(
            *((xi, yi) for xi, yi in zip(filtered_x, filtered_y) if xi not in seen and not seen.add(xi))
        )
        interpolators.append(
            interp1d(
                filtered_x,
                filtered_y,
                kind="cubic",
                fill_value=(filtered_y[0], filtered_y[-1]),
                bounds_error=False,
            )
        )
    if len(x.shape) > 1:
        return np.array([interpolator(x[:, i]) for i, interpolator in enumerate(interpolators)])
    else:
        return np.array([interpolator(x) for i, interpolator in enumerate(interpolators)])
