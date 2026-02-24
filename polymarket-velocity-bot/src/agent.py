import asyncio
from loguru import logger
from src.market_discovery import get_active_short_term_markets, get_market_book
from src.signal_generator import SignalGenerator
from src.risk_manager import RiskManager
from src.execution import TradeExecutor
from src.data_feeder import DataFeeder

class VelocityAgent:
    def __init__(self, config):
        self.config = config
        self.signal_gen = SignalGenerator()
        self.risk = RiskManager(config)
        self.feeder = DataFeeder(config["assets"])
        self.executor = TradeExecutor(config)
        self.bankroll = 10000.0

    async def run(self):
        logger.info("Velocity Engine Ignition Sequence: STARTED")
        asyncio.create_task(self.feeder.start())
        # Monitoring Task
        asyncio.create_task(self._monitoring_loop())
        
        while True:
            try:
                markets = get_active_short_term_markets(self.config["assets"], self.config["timeframes"])
                
                for m in markets:
                    asset = next((a for a in self.config["assets"] if a.lower() in m["title"].lower()), None)
                    if not asset: continue
                        
                    obs = self.feeder.get_latest(asset)
                    if obs is None: continue
                    
                    # FETCH REAL CLOB PRICE & DEPTH
                    book = get_market_book(self.executor.client, m["token_yes"]) if self.executor.client else {'mid_price': 0.50}
                    if not book: continue
                    
                    feat = {'mid_price': book['mid_price']}
                    tf = 5 if "5m" in m["title"].lower() else 15
                    
                    prob, edge = await self.signal_gen.get_edge(asset, tf, obs, feat)
                    min_edge = self.config["edge"]["min_edge"]
                    
                    side, p_bet, c_bet, token_id = None, 0, 0, None
                    if edge > min_edge:
                        side, p_bet, c_bet, token_id = "YES", prob, book['mid_price'], m["token_yes"]
                    elif edge < -min_edge:
                        side, p_bet, c_bet, token_id = "NO", 1.0 - prob, 1.0 - book['mid_price'], m["token_no"]
                    
                    if side:
                        size = self.risk.kelly_size(p=p_bet, c=c_bet, bankroll=self.bankroll)
                        if size > 1.0:
                            logger.success(f"SIGNAL: {m['title']} | {side} | Prob: {p_bet:.1%} | Prc: {c_bet:.2f} | Edge: {abs(edge):.2%}")
                            await self.executor.execute_trade(token_id, size, side, c_bet)
                
                await asyncio.sleep(15)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)

    async def _monitoring_loop(self):
        while True:
            await self.executor.monitor_and_scale()
            await asyncio.sleep(10) # Monitor every 10s
