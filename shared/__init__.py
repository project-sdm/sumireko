import math


def weight(n: int, tf: int, df: int) -> float:
    return math.log(1 + tf) * math.log((n + 1) / (df + 1))
