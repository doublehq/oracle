import logging
import sys

def configure_logging():
    """Configure logging to output to both file and console with appropriate formatting."""
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # # File handler
    # file_handler = logging.FileHandler('oracle.log')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    # root_logger.addHandler(file_handler) 