from models.gbm import GBM
from utils.util import value_at_risk, paths_to_returns



gbm = GBM(S0=100, mu=0.10, sigma=0.20, T=1.0, N=252, seed=42)
paths = gbm.simulate(paths=1000, show=True)
returns = paths_to_returns(paths)
print(value_at_risk(returns))