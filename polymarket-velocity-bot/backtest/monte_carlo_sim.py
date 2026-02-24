import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, List, Tuple
from loguru import logger
import sys

# Add root to path to import src
sys.path.append('.')
from src.models.hmm import JumpFusionHMM, Regime

class MonteCarloSimulator:
    def __init__(self, model_path="../jump_fusion_hmm.json", data_path="../BTCUSDT_1m_6mo.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.hmm = JumpFusionHMM().load_model(model_path)
        self.data = self._load_data()
        
    def _load_data(self):
        logger.info(f"Loading data for backtest: {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate HMM sensors (5m, 15m, Vol Pressure)
        df['signed_vol'] = df['volume'] * np.sign(df['close'] - df['open'])
        res_5m = df.resample('5min').agg({'close':'last','signed_vol':'sum','volume':'sum'}).ffill()
        r5 = res_5m['close'].pct_change().fillna(0)
        r15 = df['close'].resample('15min').last().ffill().pct_change().fillna(0)
        
        c = pd.DataFrame({'r5': r5, 'close': res_5m['close']}).join(r15.rename('r15')).ffill()
        c['of'] = (res_5m['signed_vol'] / (res_5m['volume'] + 1e-9)).fillna(0)
        return c.dropna()

    def run_historical_backtest(self, fee=0.007, slippage=0.002):
        """
        Runs the HMM over historical data to collect trade performance stats.
        """
        logger.info("Running historical performance scan (Vectorized)...")
        obs_vals = self.data[['r5', 'r15', 'of']].values
        returns = self.data['r5'].shift(-1).values
        
        # Vectorized Emission Probabilities
        B = self.hmm.get_emission_probs(obs_vals)
        T, N = B.shape
        
        alpha = self.hmm.pi
        trades = []
        
        for t in range(T - 1):
            # belief update
            alpha = np.dot(alpha, self.hmm.A) * B[t]
            alpha /= (np.sum(alpha) + 1e-300)
            
            regime = np.argmax(alpha)
            conf = np.max(alpha)
            
            # Predict Prob Up based on drift
            mu = self.hmm.params[regime][0]['mu']
            sigma = self.hmm.params[regime][0]['sigma']
            
            z = -mu / (sigma + 1e-9)
            prob_up_hmm = 1.0 - norm.cdf(z)
            
            # APPLY INVERSE LOGIC
            prob_up = 1.0 - prob_up_hmm
            
            # Calculate Edge
            market_price = 0.50
            edge = (prob_up - market_price)
            
            if abs(edge) > 0.005 and conf > 0.6:
                # If edge > 0, we bet YES (p=p_up, c=0.5)
                # If edge < 0, we bet NO (p=1-p_up, c=0.5)
                p_win = prob_up if prob_up > 0.5 else (1.0 - prob_up)
                c_entry = 0.50
                
                actual_dir = 1 if returns[t] > 0 else -1
                predicted_dir = 1 if prob_up > 0.5 else -1
                is_win = (actual_dir == predicted_dir)
                
                # New Binomial Kelly sizing
                raw_k = (p_win - c_entry) / (1.0 - c_entry + 1e-9)
                
                trades.append({
                    'edge': abs(edge),
                    'is_win': is_win,
                    'k_perc': raw_k
                })
        
        if trades:
            tdf = pd.DataFrame(trades)
            win_rate = tdf['is_win'].mean()
            avg_k = tdf['k_perc'].mean()
            print(f"Inverse Win Rate: {win_rate:.2%} | Avg Kelly (Raw): {avg_k:.4%}")
        
        print(f"Collected {len(trades)} historical trade samples.")
        return pd.DataFrame(trades)

    def simulate_monte_carlo(self, trade_dist: pd.DataFrame, initial_capital=10000, days=180, sims=200, kelly_fraction=0.5):
        """
        Projects equity curves based on historical trade distribution.
        """
        logger.info(f"Starting Monte Carlo Simulation ({sims} paths, {days} days)...")
        
        # trades per day (assume we find ~20 markets per day based on historical frequency)
        trades_per_day = len(trade_dist) / (len(self.data) * 5 / 1440)
        total_trades = int(trades_per_day * days)
        logger.info(f"Historical Trade Freq: {trades_per_day:.2f}/day | Total Trades for {days}d: {total_trades}")
        
        results = []
        # Pre-convert trade_dist to numpy for speed
        k_percs = trade_dist['k_perc'].values
        wins = trade_dist['is_win'].values
        
        indices = np.arange(len(trade_dist))
        
        for s in range(sims):
            if s % 50 == 0:
                logger.info(f"Simulation {s}/{sims} in progress...")
                
            # Vectorized sampling
            batch_indices = np.random.choice(indices, size=total_trades, replace=True)
            path_ks = k_percs[batch_indices]
            path_wins = wins[batch_indices]
            
            equity = [initial_capital]
            for i in range(total_trades):
                # Apply fraction and cap at 25%
                size_pct = min(path_ks[i] * kelly_fraction, 0.25)
                risk_amount = equity[-1] * size_pct
                
                if path_wins[i]:
                    equity.append(equity[-1] + risk_amount * (1.0 - 0.007)) # Win (minus fee)
                else:
                    equity.append(equity[-1] - risk_amount) # Loss
                
                if equity[-1] <= 1.0: # Effectively bankrupt
                    equity[-1] = 0
                    break
            results.append(equity)
            
        return results

    def plot_results(self, simulations, initial_capital, target=1000000):
        final_values = [p[-1] for p in simulations]
        success_rate = sum(1 for v in final_values if v >= target) / len(final_values)
        median_final = np.median(final_values)
        
        plt.figure(figsize=(12, 6))
        
        # Plot Path subset
        for i in range(min(100, len(simulations))):
            plt.plot(simulations[i], color='blue', alpha=0.1)
            
        plt.axhline(y=target, color='red', linestyle='--', label=f'Target ${target/1e6:.1f}M')
        plt.yscale('log')
        plt.title(f"Polymarket Velocity MC: 180-Day Projection (Median: ${median_final:,.0f})")
        plt.xlabel("Trade Count")
        plt.ylabel("Equity (USD Log Scale)")
        plt.legend()
        
        print(f"\n--- MONTE CARLO RESULTS ---")
        print(f"Paths Simulated: {len(simulations)}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Median Final Equity: ${median_final:,.2f}")
        print(f"Probability of hitting {target/1e6:.1f}M: {success_rate:.2%}")
        print(f"Max Simulation Equity: ${np.max(final_values):,.2f}")
        print(f"Min Simulation Equity: ${np.min(final_values):,.2f}")
        
        plt.savefig("backtest/mc_results.png")
        print(f"\nResults plot saved to backtest/mc_results.png")
