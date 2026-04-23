import numpy as np
import random
from utils.util import random_normal
import matplotlib.pyplot as plt

class GBM:

    """Geometric Brownian Motion (GBM) is a continuous-time stochastic process 
    used to model stock prices and other financial variables. It is defined
    by the stochastic differential equation:
    dS = mu * S * dt + sigma * S * dW
    """

    def __init__(self, S0: float = 100.0, mu: float = 0.10, sigma: float = 0.20, 
                 T: float = 1.0, N: int = 252, seed: int | None = None) -> None:
        #Variable definitions for Geometric Brownian Motion (GBM).

        self.S0: float    = S0     # starting price
        self.mu: float    = mu    # annual drift (10%)
        self.sigma: float = sigma    # annual volatility (20%)
        self.T: float     = T     # time horizon in years
        self.N: int       = N     # number of steps (trading days)

        self.seed: int | None = (seed if seed else random.randint(0, 2**32 - 1))  # random seed

        self.dt: float = self.T / self.N  # time step

    def __str__(self) -> str:
        return f"GBM(S0={self.S0}, mu={self.mu}, sigma={self.sigma}, T={self.T}, N={self.N})"
    
    def simulate(self, paths: int = 10, show: bool = False) -> np.ndarray:
        """Simulate paths of the Geometric Brownian Motion (GBM) process.

        Returns:
            np.ndarray: A 2D array of shape (paths, N+1) containing the simulated paths of the GBM process.
        """
        
       
        t = np.linspace(0, self.T, self.N + 1)  # time grid
        Z = random_normal(size=(paths, self.N), mean=0.0, std=1.0, seed=self.seed)  # standard normal random variables
        increment = (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z  # GBM increment with ito correction
        
        log_S = np.cumsum(increment, axis=1)
        paths_array = self.S0 * np.exp(log_S)  # convert log returns to price paths
        paths_array = np.hstack((self.S0 * np.ones((paths, 1)), paths_array))  # add initial price

        if show:
            plt.figure(figsize=(10, 6))
            for i in range(paths_array.shape[0]):
                plt.plot(paths_array[i], label=f'Path {i+1}')
            plt.title('Simulated Geometric Brownian Motion Paths')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.show()

        return paths_array
    
    def expected_return(self) -> float:
        """Calculate the expected return of the GBM process.

        Returns:
            float: The expected return over the time horizon T.
        """
        return self.mu * self.T
    
    
    