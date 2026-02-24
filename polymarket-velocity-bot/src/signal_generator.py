import torch
import pandas as pd
import numpy as np
from loguru import logger
from scipy.stats import norm
from src.models.hmm import JumpFusionHMM, Regime

class SignalGenerator:
    def __init__(self, model_path="../jump_fusion_hmm.json"):
        # Load the pre-trained Jump-Diffusion HMM
        # model is in parent dir relative to project root execution
        try:
            self.hmm = JumpFusionHMM().load_model(model_path)
            logger.info(f"Loaded HMM model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load HMM model at {model_path}: {e}. Using initialized defaults.")
            self.hmm = JumpFusionHMM()
        
        self.current_alpha = self.hmm.pi

    async def get_edge(self, asset, timeframe_minutes, observations, feat):
        """
        Calculates edge using 'Inverse Intelligence' fused with Order Flow + Vol Profile.
        observations: [r5, r15, obi, dist_to_poc]
        """
        # 1. Update HMM state belief (3 sensors used for regime detection)
        prob_vec = np.zeros(self.hmm.n_states)
        for i in range(self.hmm.n_states):
            lp = 0
            for k in range(3): # Use [r5, r15, obi]
                val = np.array([observations[k]])
                lp += self.hmm.fusion_weights[i, k] * np.log(self.hmm.sensor_pdf(val, i, k)[0])
            prob_vec[i] = np.exp(max(lp, -700))
            
        self.current_alpha = np.dot(self.current_alpha, self.hmm.A) * prob_vec
        self.current_alpha /= (np.sum(self.current_alpha) + 1e-300)
        
        regime = np.argmax(self.current_alpha)
        conf = np.max(self.current_alpha)
        
        # 2. Derive HMM Predicted Probability
        mu_5m = self.hmm.params[regime][0]['mu']
        sigma_5m = self.hmm.params[regime][0]['sigma']
        time_scale = timeframe_minutes / 5.0
        mu_target = mu_5m * time_scale
        sigma_target = sigma_5m * np.sqrt(time_scale)
        
        z_score = -mu_target / (sigma_target + 1e-9)
        prob_up_hmm = 1.0 - norm.cdf(z_score)
        
        # 3. APPLY INVERSE LOGIC (Mean Reversion)
        prob_up = 1.0 - prob_up_hmm
        
        # 4. Integrate Volume Profile Magnet
        dist_to_poc = observations[3]
        # If we are betting UP (prob_up > 0.5) but we are far ABOVE POC (dist > 0),
        # the price is "expensive", we damp the edge.
        # If we are BELOW POC (dist < 0) and betting UP, we amplify.
        vp_multiplier = 1.0 - (dist_to_poc * 5) # 5x sensitivity to POC distance
        vp_multiplier = np.clip(vp_multiplier, 0.5, 1.5)
        
        # 5. Calculate Edge vs Polymarket Price
        mid_price = feat.get('mid_price', 0.50)
        edge = (prob_up - mid_price) * vp_multiplier
        
        if conf < 0.6:
            edge *= 0.1
            
        return prob_up, edge
