import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

class Regime(IntEnum):
    TRENDING = 0
    MEAN_REVERTING = 1
    JUMP = 2

@dataclass
class HMMParams:
    n_regimes: int = 3
    n_sensors: int = 3
    jump_components: int = 2
    robust_threshold: float = 3.0

class JumpFusionHMM:
    def __init__(self, config: HMMParams = None):
        self.config = config or HMMParams()
        self.n_states = self.config.n_regimes
        self.n_sensors = self.config.n_sensors
        
        self.A = np.array([
            [0.95, 0.03, 0.02],
            [0.05, 0.90, 0.05],
            [0.10, 0.10, 0.80]
        ])
        self.pi = np.ones(self.n_states) / self.n_states
        self.params = self._init_params()
        self.fusion_weights = np.ones((self.n_states, self.n_sensors)) / self.n_sensors

    def _init_params(self):
        params = {}
        for r in range(self.n_states):
            params[r] = {}
            for s in range(self.n_sensors):
                params[r][s] = {
                    'mu': 0.0,
                    'sigma': 0.01 if s < 2 else 0.5,
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
        diff = norm.pdf(x, p['mu'], sigma)
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

    def load_model(self, path: str):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.A = np.array(data['A'])
            # Convert string keys back to int for params
            self.params = {int(k): {int(sk): sv for sk, sv in v.items()} for k, v in data['params'].items()}
        return self
