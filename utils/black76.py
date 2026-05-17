import math
from math import erf, exp, log, sqrt
from statistics import NormalDist

norm = NormalDist()
_EPS = 1e-12


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def black76_call(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return exp(-r*T) * max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))


def black76_put(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return exp(-r*T) * max(K - F, 0.0)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def implied_vol_black76(price, F, K, T, r, option_type="call", tol=1e-6, max_iter=100):
    sigma_low, sigma_high = 1e-6, 3.0  # границы для поиска (0% .. 300%)
    
    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        model_price = (black76_call(F, K, T, r, sigma_mid)
                       if option_type == "call"
                       else black76_put(F, K, T, r, sigma_mid))
        
        if abs(model_price - price) < tol:
            return sigma_mid
        
        if model_price > price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    
    return sigma_mid / 100


def black76_put_delta(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        # В пределе к экспирации дельта пута = -1, если F < K, и 0, если F > K
        if F < K:
            return -math.exp(-r * max(T, 0.0))
        else:
            return 0.0

    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return -math.exp(-r * T) * norm.cdf(-d1)


def black76_call_delta(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        # В пределе к экспирации дельта колла = 1, если F > K, и 0, если F < K
        if F > K:
            return math.exp(-r * max(T, 0.0))
        else:
            return 0.0

    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * norm.cdf(d1)


def _d1_d2(F: float, K: float, T: float, sigma: float):
    T = max(T, _EPS)
    sigma = max(sigma, _EPS)
    vol_sqrt = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2


def black76_call_theta(F: float, K: float, T: float, r: float, sigma: float) -> float:
    T = max(T, _EPS)
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = exp(-r * T)

    price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    term_time = df * (F * norm.pdf(d1) * sigma / (2.0 * sqrt(T)))
    term_rate = -r * price
    return term_time + term_rate

def black76_put_theta(F: float, K: float, T: float, r: float, sigma: float) -> float:
    T = max(T, _EPS)
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = exp(-r * T)

    price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    term_time = df * (F * norm.pdf(d1) * sigma / (2.0 * sqrt(T)))
    term_rate = -r * price
    return term_time + term_rate


def black76_gamma(F: float, K: float, T: float, r: float, sigma: float) -> float:
    T = max(T, _EPS)
    sigma = max(sigma, _EPS)
    d1, _ = _d1_d2(F, K, T, sigma)
    return math.exp(-r * T) * norm.pdf(d1) / (F * sigma * math.sqrt(T))


def black76_vega(F: float, K: float, T: float, r: float, sigma: float) -> float:
    T = max(T, _EPS)
    d1, _ = _d1_d2(F, K, T, sigma)
    return math.exp(-r * T) * F * norm.pdf(d1) * math.sqrt(T)


def black76_vega_1vol(F: float, K: float, T: float, r: float, sigma: float) -> float:
    return black76_vega(F, K, T, r, sigma) / 100.0


