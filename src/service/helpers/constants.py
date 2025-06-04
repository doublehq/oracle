import logging
from .logging_config import configure_logging

# Configure logging
configure_logging()

# Create logger
logger = logging.getLogger()

CASH_CUSIP_ID = "_CASH_123"
CASH_SYMBOL = "_CASH_"