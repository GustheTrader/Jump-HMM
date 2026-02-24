# Jump-Diffusion HMM with Multi-Sensor Fusion

A robust, high-performance Hidden Markov Model (HMM) designed for regime detection in cryptocurrency markets. This implementation features Jump-Diffusion emission modeling and multi-sensor data fusion to capture both smooth trending behavior and sudden, discrete volatility spikes (jumps).

## Features

- **Jump-Diffusion Modeling**: Accounts for the "fat-tails" in financial data by mixing Gaussian diffusion with discrete jump components.
- **Multi-Sensor Fusion**: Integrates multiple inputs (e.g., 5m returns, 15m returns, and Volume Delta) using a weighted fusion approach in log-space.
- **Robust Training**: Optimized Baum-Welch algorithm with numerical safeguards (log-sum-exp trick, probability floors) to prevent NaN errors and ensure convergence on real-world data.
- **Lookahead-Corrected Backtester**: Realistic trading simulation that handles transaction costs and proper signal-to-execution timing.

## Installation

```bash
pip install numpy pandas scipy ccxt
```

## Usage

1. **Fetch Data**: Use `fetch_6m_data.py` to get high-resolution Binance OHLCV data.
2. **Train Model**: Run `JumpHmm.py` to train the HMM on the historical data and generate the `jump_fusion_hmm.json` model file.
3. **Analyze Results**: The script will output training log-likelihoods and a final backtest return summary.

## Files

- `JumpHmm.py`: Core HMM engine, training logic, and strategy runner.
- `fetch_6m_data.py`: Multi-month 1-minute data downloader for Binance.
- `jump_fusion_hmm.json`: Weights and parameters for the trained regimes.
