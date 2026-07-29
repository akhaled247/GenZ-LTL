import numpy as np


def average_reward_per_step(returns: list[float], num_frames: list[int]) -> float:
    if len(returns) != len(num_frames):
        raise ValueError("The length of the returns and num_frames lists must be the same.")
    avgs = []
    for i in range(len(returns)):
        if num_frames[i] > 0:
            avgs.append(returns[i] / num_frames[i])
    if not avgs:
        return 0.0
    return float(np.mean(avgs))


def average_discounted_return(returns: list[float], num_frames: list[int], discount: float):
    if len(returns) != len(num_frames):
        raise ValueError("The length of the returns and num_frames lists must be the same.")
    discounted_returns = []
    for i in range(len(returns)):
        if num_frames[i] > 0:
            discounted_returns.append(returns[i] * (discount ** (num_frames[i] - 1)))
    if not discounted_returns:
        return 0.0
    return float(np.mean(discounted_returns))
