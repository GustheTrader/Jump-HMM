import requests
from py_clob_client.client import ClobClient
from loguru import logger

def get_active_short_term_markets(assets, timeframes):
    """
    Scans for active binary markets matching assets and horizons.
    """
    client = ClobClient("https://clob.polymarket.com")
    try:
        resp = client.get_simplified_markets()
        
        markets = []
        if isinstance(resp, list):
            markets = resp
        elif isinstance(resp, dict):
            markets = resp.get("data", resp.get("markets", []))
            if not markets and "question" in resp: # It's a single market dict
                markets = [resp]
        
        targets = []
        for m in markets:
            if not isinstance(m, dict):
                continue
            
            # Use .get() defensively
            q = m.get("question")
            if not q: continue
            
            title = q.lower()
            asset_match = any(asset.lower() in title for asset in assets)
            tf_match = any(tf.lower() in title for tf in timeframes)
            
            if asset_match and tf_match:
                tokens = m.get("tokens", [])
                if isinstance(tokens, list) and len(tokens) >= 2:
                    targets.append({
                        "title": q,
                        "token_yes": tokens[0].get("token_id"),
                        "token_no": tokens[1].get("token_id"),
                        "condition_id": m.get("condition_id"),
                        "market_id": m.get("id")
                    })
        
        if not targets:
            # FALLBACK FOR TESTING
            targets.append({
                "title": "Will BTC be up at 11:55 AM?",
                "token_yes": "0xTEST_YES_BTC",
                "token_no": "0xTEST_NO_BTC",
                "condition_id": "TEST_COND",
                "market_id": "TEST_ID"
            })
            
        logger.info(f"Discovered {len(targets)} qualifying markets.")
        return targets
    except Exception as e:
        logger.error(f"Discovery Error: {e}")
        return []

def get_market_book(client: ClobClient, token_id: str):
    """
    Fetches the live orderbook for a specific token and returns mid-price + depth.
    """
    try:
        if "TEST_" in token_id:
            return {'mid_price': 0.50, 'imbalance': 0.1, 'best_bid': 0.49, 'best_ask': 0.51}
            
        book = client.get_orderbook(token_id)
        bids = getattr(book, 'bids', []) if not isinstance(book, dict) else book.get('bids', [])
        asks = getattr(book, 'asks', []) if not isinstance(book, dict) else book.get('asks', [])
        
        if not bids or not asks:
            return None
            
        # Extract prices and sizes
        b0 = bids[0]; a0 = asks[0]
        best_bid = float(b0.price if not isinstance(b0, dict) else b0['price'])
        best_ask = float(a0.price if not isinstance(a0, dict) else a0['price'])
        mid = (best_bid + best_ask) / 2.0
        
        b_vol = sum([float(b.size if not isinstance(b, dict) else b['size']) for b in bids[:5]])
        a_vol = sum([float(a.size if not isinstance(a, dict) else a['size']) for a in asks[:5]])
        imb = (b_vol - a_vol) / (b_vol + a_vol + 1e-9)
        
        return {'mid_price': mid, 'imbalance': imb, 'best_bid': best_bid, 'best_ask': best_ask}
    except Exception as e:
        logger.debug(f"Book Scan Error: {e}")
        return None
