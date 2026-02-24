from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import OrderArgs
from loguru import logger

class TradeExecutor:
    def __init__(self, config):
        self.config = config
        self.client = self._init_client()
        self.active_positions = {} # token_id -> {qty, entry_price, side}

    def _init_client(self):
        logger.info("Initializing Polymarket CLOB Client...")
        # For dry-run/test without key, handle missing key gracefully
        pk = self.config["polymarket"].get("private_key")
        if not pk or "YOUR_HEX" in pk:
            logger.warning("No Private Key found. Execution in DRY-RUN mode.")
            return None
        return ClobClient(
            host=self.config["polymarket"]["host"],
            key=pk,
            chain_id=POLYGON,
            funder=self.config["polymarket"].get("funder", "")
        )

    async def execute_trade(self, token_id, size_usd, side, price):
        qty = round(size_usd / (price + 1e-9), 2)
        logger.info(f"PLANNING {side} | Qty: {qty} | Price: {price}")
        
        if self.client:
            try:
                order_args = OrderArgs(price=price, size=qty, side="BUY", token_id=token_id)
                signed_order = self.client.create_order(order_args)
                resp = self.client.post_order(signed_order)
                if resp.get("success"):
                    logger.success(f"LIVE TRADE PLACED: {resp.get('orderID')}")
                    self.active_positions[token_id] = {'qty': qty, 'entry': price, 'side': side, 'scaled': False}
                    return resp.get("orderID")
            except Exception as e:
                logger.error(f"Execution failed: {e}")
        else:
            # Dry-run tracking
            logger.info(f"DRY-RUN: Position logged for {token_id}")
            self.active_positions[token_id] = {'qty': qty, 'entry': price, 'side': side, 'scaled': False}
            return "DRY_RUN_ID"

    async def monitor_and_scale(self):
        """
        Monitors active positions and scales out 50% at 50% profit.
        Example: Entry 0.50 -> 50% profit is at 0.75.
        """
        for token_id, pos in list(self.active_positions.items()):
            if pos['scaled']: continue
            
            # Fetch current price
            try:
                book = self.client.get_orderbook(token_id) if self.client else None
                mid = 0.60 # Dummy for dry-run testing
                if book:
                    mid = (float(book.bids[0].price) + float(book.asks[0].price)) / 2.0
                
                # Calculate ROI
                profit_pct = (mid - pos['entry']) / pos['entry']
                if profit_pct >= 0.50:
                    scale_qty = round(pos['qty'] / 2, 2)
                    logger.warning(f"SCALE OUT: 50% profit hit for {pos['side']}! Selling {scale_qty} at {mid}")
                    
                    if self.client:
                        order_args = OrderArgs(price=mid, size=scale_qty, side="SELL", token_id=token_id)
                        signed_order = self.client.create_order(order_args)
                        self.client.post_order(signed_order)
                    
                    pos['scaled'] = True
                    pos['qty'] -= scale_qty
            except Exception as e:
                logger.error(f"Monitoring error for {token_id}: {e}")
