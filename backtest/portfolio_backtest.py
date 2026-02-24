import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from loguru import logger
import sys

# Add project root to path
sys.path.append('polymarket-velocity-bot')
from src.models.hmm import JumpFusionHMM

class PortfolioBacktester:
    def __init__(self, 
                 assets=['BTC', 'ETH', 'SOL', 'XRP'], 
                 model_path="jump_fusion_hmm.json",
                 data_prefix=""):
        self.assets = assets
        self.hmm = JumpFusionHMM().load_model(model_path)
        self.data_prefix = data_prefix
        self.cap = 10000.0
        self.fee = 0.007
        self.kelly_fraction = 0.5
        
    def load_asset_data(self, asset):
        filename = f"{self.data_prefix}{asset}USDT_1m_6mo.csv"
        logger.info(f"Loading {filename}...")
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        df['signed_vol'] = df['volume'] * np.sign(df['close'] - df['open'])
        
        res_5m = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'signed_vol': 'sum'
        }).ffill()
        
        res_5m['r5'] = res_5m['close'].pct_change(1).fillna(0)
        res_5m['r15'] = res_5m['close'].pct_change(3).fillna(0)
        res_5m['of'] = (res_5m['signed_vol'] / (res_5m['volume'] + 1e-9)).fillna(0)
        
        return df, res_5m

    def run(self, days=180):
        # 1. First pass: load and find common index
        temp_data = {}
        for asset in self.assets:
            try:
                raw, res = self.load_asset_data(asset)
                temp_data[asset] = (raw, res)
            except Exception as e:
                logger.error(f"Failed to load {asset}: {e}")
        
        if 'BTC' not in temp_data:
            return None, None
            
        common_idx = temp_data['BTC'][1].index
        if days:
            start_date = common_idx[-1] - pd.Timedelta(days=days)
            common_idx = common_idx[common_idx >= start_date]

        # 2. Re-align all data to common index
        portfolio_data = {}
        for asset, (raw, res) in temp_data.items():
            # Align resampled data
            res_aligned = res.reindex(common_idx).ffill().fillna(0)
            obs = res_aligned[['r5', 'r15', 'of']].values
            B = self.hmm.get_emission_probs(obs)
            
            # Map 1m intraday data
            df_1m = raw.assign(idx_5m = lambda x: (x.index.values.astype('int64') // (5 * 60 * 10**9)))
            groups = df_1m.groupby('idx_5m')
            intraday_data = {t: (g['high'].values, g['low'].values, g['close'].iloc[0], g['close'].iloc[-1]) 
                             for t, g in groups}
            
            portfolio_data[asset] = {
                'emissions': B,
                'intraday': intraday_data
            }

        logger.info(f"Starting optimized backtest for {len(common_idx)} periods...")
        
        equity = [self.cap]
        trades = []
        alphas = {asset: self.hmm.pi for asset in self.assets}
        
        for i in range(len(common_idx) - 1):
            curr_time = common_idx[i]
            t_int = curr_time.value // (5 * 60 * 10**9)
            
            for asset in self.assets:
                data = portfolio_data.get(asset)
                if not data: continue
                
                # Update HMM
                B_val = data['emissions'][i]
                alphas[asset] = np.dot(alphas[asset], self.hmm.A) * B_val
                alphas[asset] /= (np.sum(alphas[asset]) + 1e-300)
                
                regime = np.argmax(alphas[asset])
                conf = np.max(alphas[asset])
                
                mu = self.hmm.params[regime][0]['mu']
                sigma = self.hmm.params[regime][0]['sigma']
                # Inverse Logic check
                z = -mu / (sigma + 1e-9)
                prob_up_hmm = 1.0 - norm.cdf(z)
                prob_up = 1.0 - prob_up_hmm
                
                edge = prob_up - 0.50
                
                if abs(edge) > 0.005 and conf > 0.6:
                    side = "YES" if edge > 0 else "NO"
                    p_win = prob_up if side == "YES" else (1.0 - prob_up)
                    c_entry = 0.50
                    
                    raw_k = (p_win - c_entry) / (1.0 - c_entry + 1e-9)
                    size_pct = min(raw_k * self.kelly_fraction, 0.25)
                    risk_amount = equity[-1] * size_pct
                    
                    if risk_amount < 1.0: continue
                    
                    intra = data['intraday'].get(t_int)
                    if not intra: continue
                    
                    highs, lows, entry_px, exit_px = intra
                    scaled_out = False
                    move_threshold = 0.0015
                    
                    if side == "YES":
                        if np.any((highs - entry_px) / entry_px >= move_threshold):
                            scaled_out = True
                    else:
                        if np.any((lows - entry_px) / entry_px <= -move_threshold):
                            scaled_out = True
                    
                    is_win = (exit_px > entry_px) if side == "YES" else (exit_px < entry_px)
                    
                    if scaled_out:
                        locked_pnl = 0.5 * (risk_amount * 0.5) 
                        settle_pnl = 0.5 * (risk_amount * (1.0 - self.fee)) if is_win else 0.5 * (-risk_amount)
                        trade_pnl = locked_pnl + settle_pnl
                    else:
                        trade_pnl = risk_amount * (1.0 - self.fee) if is_win else -risk_amount
                    
                    equity[-1] += trade_pnl
                    trades.append({
                        'timestamp': curr_time,
                        'asset': asset,
                        'side': side,
                        'size': risk_amount,
                        'scaled': scaled_out,
                        'is_win': is_win,
                        'pnl': trade_pnl
                    })
            equity.append(equity[-1])

        tdf = pd.DataFrame(trades)
        equity = equity[:len(common_idx)]
        final_equity = equity[-1]
        
        print(f"\n--- SCRUBBED PORTFOLIO BACKTEST RESULTS ---")
        print(f"Final Equity  : ${final_equity:,.2f}")
        print(f"Total Trades  : {len(tdf)}")
        if not tdf.empty:
            print(f"Win Rate      : {tdf['is_win'].mean():.2%}")
            print(f"Scale-outs    : {tdf['scaled'].sum()} ({tdf['scaled'].mean():.1%})")
            p_sum = tdf[tdf['pnl']>0]['pnl'].sum()
            l_sum = abs(tdf[tdf['pnl']<0]['pnl'].sum())
            print(f"Profit Factor : {p_sum / (l_sum + 1e-9):.2f}")
        
        print(f"Alignment check: Equity len {len(equity)}, Index len {len(common_idx)}")
        plt_df = pd.DataFrame({'equity': equity}, index=common_idx)
        return plt_df, tdf

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tester = PortfolioBacktester()
    plt_df, tdf = tester.run(days=180)
    
    if plt_df is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(plt_df['equity'], color='forestgreen', linewidth=1.5)
        plt.yscale('log')
        plt.title("6-Month Portfolio Velocity Backtest (BTC+ETH+SOL+XRP)\nScale-Out 1/2 @ 50% ROI enabled")
        plt.ylabel("USD (Log Scale)")
        plt.grid(True, alpha=0.3)
        plt.savefig("backtest/portfolio_results.png")
        print("\nPlot saved to backtest/portfolio_results.png")
