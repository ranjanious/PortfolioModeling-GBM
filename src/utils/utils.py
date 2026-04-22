import numpy as np

def normal(size: int, mean: float = 0.0, std: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Generate a random number from a normal distribution.

    Args:
        size (int): The size of the output array.
        mean (float, optional): The mean of the normal distribution. Defaults to 0.0.
        std (float, optional): The standard deviation of the normal distribution. Defaults to 1.0.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: An array of random numbers drawn from a normal distribution with mean `mean` and standard deviation `std`.
    """
    
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=size)


def uniform(size: int, range: tuple[float, float] = (0.0, 1.0), seed: int | None = None) -> np.ndarray:
    """Generate a random number from a uniform distribution.

    Args:
        size (int): The size of the output array.
        range (tuple(float, float), optional): The range of the uniform distribution. Defaults to (0.0, 1.0).
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: An array of random numbers drawn from a uniform distribution over the specified range.
    """
    
    rng = np.random.default_rng(seed)
    return rng.uniform(low=range[0], high=range[1], size=size)

def bernoulli(size: int, p: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Generate a random number from a Bernoulli distribution.

    Args:
        size (int): The size of the output array.
        p (float, optional): The probability of success. Defaults to 0.5.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: An array of random numbers drawn from a Bernoulli distribution with probability `p`.
    """
    
    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=p, size=size)

