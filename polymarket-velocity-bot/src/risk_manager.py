from loguru import logger

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.kelly_fraction = config["risk"].get("kelly_fraction", 0.5)
        self.max_dd = config["risk"].get("max_dd", 0.20)
        
    def kelly_size(self, p, c, bankroll):
        """
        Calculates optimal fractional Kelly size for a binomial market ($1 payout).
        Formula: f* = (p - c) / (1 - c)
        p: Win probability [0, 1]
        c: Entry price [0, 1]
        """
        # Ensure we are betting with edge
        if p <= c:
            return 0
        
        # Binomial Kelly for payout of 1.0
        raw_kelly = (p - c) / (1.0 - c + 1e-9)
        
        # Apply fractional multiplier
        safe_kelly = raw_kelly * self.kelly_fraction
        
        # Cap at max risk per trade (e.g., 25%)
        capped_kelly = min(safe_kelly, self.config["edge"]["max_risk_per_trade"])
        
        # Safety check for extreme bankroll situations
        if capped_kelly < 0: return 0
        
        size = bankroll * capped_kelly
        logger.info(f"Binomial Kelly: p={p:.2%}, c={c:.2f} | Size: ${size:.2f} ({capped_kelly:.2%})")
        return size

    async def update_balance(self):
        # TODO: Integrate with wallet/contract to get real USDC balance
        return 10000.0 
