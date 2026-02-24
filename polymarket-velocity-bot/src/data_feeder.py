import asyncio
import pandas as pd
import numpy as np
from binance import AsyncClient, BinanceSocketManager
from loguru import logger
from datetime import datetime

class DataFeeder:
    def __init__(self, assets: list):
        self.assets = [f"{a}USDT" for a in assets]
        self.buffers = {asset: [] for asset in self.assets}
        self.latest_sensors = {asset: None for asset in self.assets}
        self.orderbooks = {asset: {'bids': np.zeros((10, 2)), 'asks': np.zeros((10, 2))} for asset in self.assets}
        self.vprofile = {asset: {} for asset in self.assets} # Price bin -> Cumulative Volume
        self.client = None

    async def start(self):
        # Switching to binance.us for US-based terminal tests
        self.client = await AsyncClient.create(tld='us')
        bm = BinanceSocketManager(self.client)
        
        tasks = []
        for asset in self.assets:
            tasks.append(self._handle_kline(bm.kline_socket(symbol=asset, interval='1m'), asset))
            tasks.append(self._handle_depth(bm.depth_socket(symbol=asset, depth='10'), asset))
            tasks.append(self._handle_trades(bm.trade_socket(symbol=asset), asset))
        
        logger.info(f"Igniting High-Velocity Streams: OBI + VolProfile for {self.assets}")
        await asyncio.gather(*tasks)

    async def _handle_kline(self, socket, asset):
        async with socket as stream:
            while True:
                res = await stream.recv()
                if res['e'] == 'kline' and res['k']['x']:
                    self._process_bar(asset, res['k'])

    async def _handle_depth(self, socket, asset):
        async with socket as stream:
            while True:
                res = await stream.recv()
                # Update top 10 levels from Binance.us depth payload
                if 'b' in res:
                    self.orderbooks[asset]['bids'] = np.array(res['b'][:10], dtype=float)
                if 'a' in res:
                    self.orderbooks[asset]['asks'] = np.array(res['a'][:10], dtype=float)

    async def _handle_trades(self, socket, asset):
        async with socket as stream:
            while True:
                res = await stream.recv()
                # Update Volume Profile (Binning to 0.05% intervals)
                price = float(res['p'])
                vol = float(res['q'])
                v_bin = round(price * 2000) / 2000 # 5bps bins
                self.vprofile[asset][v_bin] = self.vprofile[asset].get(v_bin, 0) + vol

    def _process_bar(self, asset, k):
        bar = {
            'timestamp': datetime.fromtimestamp(k['t']/1000),
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v'])
        }
        self.buffers[asset].append(bar)
        if len(self.buffers[asset]) > 100:
            self.buffers[asset].pop(0)
            
        if len(self.buffers[asset]) >= 15:
            self.latest_sensors[asset] = self._calculate_sensors(asset)

    def _calculate_sensors(self, asset):
        df = pd.DataFrame(self.buffers[asset])
        
        # 1. Base HMM Sensors
        r5 = (df['close'].iloc[-1] / df['close'].iloc[-5]) - 1
        r15 = (df['close'].iloc[-1] / df['close'].iloc[-15]) - 1
        
        # 2. Order Book Imbalance (OBI)
        # (Bid_Vol - Ask_Vol) / (Total_Vol) for top 5 levels
        bids = self.orderbooks[asset]['bids'][:5, 1].sum()
        asks = self.orderbooks[asset]['asks'][:5, 1].sum()
        obi = (bids - asks) / (bids + asks + 1e-9)
        
        # 3. Volume Profile (Distance to Point of Control)
        # POC = bin with max volume
        if self.vprofile[asset]:
            poc_price = max(self.vprofile[asset], key=self.vprofile[asset].get)
            curr_price = df['close'].iloc[-1]
            dist_to_poc = (curr_price - poc_price) / curr_price
        else:
            dist_to_poc = 0
            
        return [r5, r15, obi, dist_to_poc]

    def get_latest(self, asset):
        # Allow both "BTC" and "BTCUSDT" lookups
        key = asset if asset.endswith("USDT") else f"{asset}USDT"
        return self.latest_sensors.get(key)
