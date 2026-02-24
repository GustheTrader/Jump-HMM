from src.agent import VelocityAgent
import yaml
import asyncio
from loguru import logger
import sys

# Configure Loguru
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("logs/bot.log", rotation="10 MB")

async def main():
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
            
        agent = VelocityAgent(config)
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
