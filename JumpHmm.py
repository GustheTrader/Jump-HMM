import numpy as np
import pandas as pd
from scipy.stats import norm
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

class Regime(IntEnum):
    TRENDING = 0
    MEAN_REVERTING = 1
    JUMP = 2

@dataclass
class FusionConfig:
    weights: Dict[str, Dict[str, float]] = None
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'trending': {'r5': 0.5, 'r15': 0.3, 'of': 0.2},
                'mean_reverting': {'r5': 0.2, 'r15': 0.5, 'of': 0.3},
                'jump': {'r5': 0.6, 'r15': 0.1, 'of': 0.3}
            }

@dataclass
class HMMParams:
    n_regimes: int = 3
    n_sensors: int = 3
    jump_components: int = 2
    robust_threshold: float = 3.0

# ============================================================================
# CORE JUMP-DIFFUSION HMM (MULTI-SENSOR)
# ============================================================================

class JumpFusionHMM:
    def __init__(self, config: HMMParams = None):
        self.config = config or HMMParams()
        self.n_states = self.config.n_regimes
        self.n_sensors = self.config.n_sensors
        
        # Transition Matrix
        self.A = np.array([
            [0.95, 0.03, 0.02],
            [0.05, 0.90, 0.05],
            [0.10, 0.10, 0.80]
        ])
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Emission parameters: self.params[regime][sensor] = {mu, sigma, ...}
        self.params = self._init_params()
        
        # Fusion weights: [regime, sensor]
        self.fusion_weights = np.ones((self.n_states, self.n_sensors)) / self.n_sensors

    def _init_params(self):
        params = {}
        for r in range(self.n_states):
            params[r] = {}
            for s in range(self.n_sensors):
                params[r][s] = {
                    'mu': 0.0,
                    'sigma': 0.01 if s < 2 else 0.5, # Returns vs OrderFlow
                    'jump_lambda': 0.05,
                    'jump_mu': [0.02, -0.02],
                    'jump_sigma': [0.05, 0.05],
                    'mixture_weights': [0.5, 0.5]
                }
        return params

    def sensor_pdf(self, x: np.ndarray, regime: int, sensor: int) -> np.ndarray:
        p = self.params[regime][sensor]
        l = p['jump_lambda']
        sigma = max(p['sigma'], 1e-6)
        
        # Diffusion
        diff = norm.pdf(x, p['mu'], sigma)
        
        # Jumps
        jm = np.zeros_like(x)
        for k in range(self.config.jump_components):
            js = max(p['jump_sigma'][k], 1e-6)
            jm += p['mixture_weights'][k] * norm.pdf(x, p['jump_mu'][k], js)
            
        return np.clip((1-l)*diff + l*jm, 1e-300, 1e100)

    def get_emission_probs(self, obs: np.ndarray) -> np.ndarray:
        T, NS = obs.shape
        B = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            log_prob = np.zeros(T)
            for k in range(NS):
                # Weighted fusion in log-space
                log_prob += self.fusion_weights[i, k] * np.log(self.sensor_pdf(obs[:, k], i, k))
            B[:, i] = np.exp(np.clip(log_prob, -700, 700))
        return B

    def forward_backward(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        T, N = B.shape
        alpha = np.zeros((T, N))
        c = np.zeros(T)
        
        alpha[0] = self.pi * B[0]
        s = np.sum(alpha[0])
        if s < 1e-300: alpha[0] = 1.0/N; s = 1.0
        c[0] = 1.0/s
        alpha[0] *= c[0]
        
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t-1], self.A) * B[t]
            s = np.sum(alpha[t])
            if s < 1e-300: alpha[t] = 1.0/N; s = 1.0
            c[t] = 1.0/s
            alpha[t] *= c[t]
            
        beta = np.zeros((T, N))
        beta[-1] = c[-1]
        for t in range(T-2, -1, -1):
            beta[t] = np.dot(self.A, B[t+1] * beta[t+1]) * c[t]
            beta[t] = np.clip(beta[t], 1e-300, 1e100)
            
        gamma = alpha * beta
        gamma /= (np.sum(gamma, axis=1, keepdims=True) + 1e-300)
        return alpha, beta, gamma, -np.sum(np.log(c))

# ============================================================================
# TRAINER (FULL MULTI-SENSOR UPDATE)
# ============================================================================

class RobustBaumWelch:
    def __init__(self, hmm: JumpFusionHMM, max_iter: int = 30, tol: float = 1e-4):
        self.hmm = hmm
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, obs: np.ndarray):
        T, NS = obs.shape
        prev_ll = -np.inf
        
        for i in range(self.max_iter):
            B = self.hmm.get_emission_probs(obs)
            alpha, beta, gamma, ll = self.hmm.forward_backward(B)
            
            # Update A
            xi_sum = (alpha[:-1].T @ (B[1:] * beta[1:])) * self.hmm.A
            self.hmm.A = xi_sum / (np.sum(xi_sum, axis=1, keepdims=True) + 1e-300)
            self.hmm.A = (self.hmm.A + 1e-9) / (1.0 + 1e-9 * self.hmm.n_states)
            self.hmm.pi = gamma[0] / (np.sum(gamma[0]) + 1e-300)
            
            # Update ALL sensors for EACH regime
            for r in range(self.hmm.n_states):
                w = gamma[:, r]
                tw = max(np.sum(w), 1e-9)
                for s in range(NS):
                    val = obs[:, s]
                    mu = np.sum(w * val) / tw
                    var = np.sum(w * (val - mu)**2) / tw
                    
                    self.hmm.params[r][s]['mu'] = mu
                    self.hmm.params[r][s]['sigma'] = np.sqrt(max(var, 1e-9))
                    
                    # Jump Update (Only for return sensors)
                    if s < 2:
                        res = np.abs(val - mu)
                        mask = res > (self.hmm.config.robust_threshold * self.hmm.params[r][s]['sigma'])
                        self.hmm.params[r][s]['jump_lambda'] = np.clip(np.sum(w * mask) / tw, 0.01, 0.5)

            print(f"Iter {i}: LL = {ll:.2f}", flush=True)
            if abs(ll - prev_ll) < self.tol: break
            prev_ll = ll
        return self.hmm

# ============================================================================
# REFINED STRATEGY
# ============================================================================

class RegimeAwareStrategy:
    def __init__(self, hmm: JumpFusionHMM):
        self.hmm = hmm
        self.alpha = hmm.pi
        
    def generate_signal(self, obs: np.ndarray, feat: Dict) -> Dict:
        prob_vec = np.zeros(self.hmm.n_states)
        for i in range(self.hmm.n_states):
            lp = 0
            for k in range(self.hmm.n_sensors):
                lp += self.hmm.fusion_weights[i, k] * np.log(self.hmm.sensor_pdf(np.array([obs[k]]), i, k)[0])
            prob_vec[i] = np.exp(max(lp, -700))
            
        self.alpha = np.dot(self.alpha, self.hmm.A) * prob_vec
        self.alpha /= (np.sum(self.alpha) + 1e-300)
        
        regime = np.argmax(self.alpha)
        conf = np.max(self.alpha)
        sig = 0
        
        # Dynamic strategy based on learned regime properties
        # We look at the drift (mu) of sensor 0 (5m returns)
        drift = self.hmm.params[regime][0]['mu']
        vol = self.hmm.params[regime][0]['sigma']
        
        if drift > 0.0001 and conf > 0.6: sig = 1
        elif drift < -0.0001 and conf > 0.6: sig = -1
        
        # Mean reversion filter
        if abs(feat.get('z',0)) > 2.0: sig = -np.sign(feat.get('z',0))
            
        return {'regime': regime, 'signal': sig, 'prob': float(conf)}

def load_and_prep_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['signed_vol'] = df['volume'] * np.sign(df['close'] - df['open'])
    res_5m = df.resample('5min').agg({'close':'last','signed_vol':'sum','volume':'sum'}).ffill()
    r5 = res_5m['close'].pct_change().fillna(0)
    r15 = df['close'].resample('15min').last().ffill().pct_change().fillna(0)
    c = pd.DataFrame({'r5': r5, 'close': res_5m['close']}).join(r15.rename('r15')).ffill()
    c['of'] = (res_5m['signed_vol'] / (res_5m['volume'] + 1e-9)).fillna(0)
    return c.dropna()

if __name__ == "__main__":
    np.random.seed(42)
    data = load_and_prep_data('BTCUSDT_1m_6mo.csv')
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    
    hmm = JumpFusionHMM()
    trainer = RobustBaumWelch(hmm, max_iter=30)
    trained = trainer.fit(train[['r5', 'r15', 'of']].values)
    
    print("\nBacktesting (Lookahead Corrected)...")
    strat = RegimeAwareStrategy(trained)
    equity = [1.0]
    z = (test['r15'] - test['r15'].rolling(20).mean()) / (test['r15'].rolling(20).std() + 1e-9)
    obs_test = test[['r5', 'r15', 'of']].values
    returns_test = test['r5'].values
    pos = 0
    
    for t in range(20, len(test)-1):
        res = strat.generate_signal(obs_test[t], {'z': z.iloc[t]})
        new_pos = res['signal']
        cost = abs(new_pos - pos) * 0.0005
        equity.append(equity[-1] * (1 + new_pos * returns_test[t+1] - cost))
        pos = new_pos
        
    print(f"\nFinal Return: {equity[-1]-1:.4%}")
    with open('jump_fusion_hmm.json', 'w') as f:
        # Custom encoder for dict of dicts
        json.dump({'A': trained.A.tolist(), 'params': trained.params}, f)
