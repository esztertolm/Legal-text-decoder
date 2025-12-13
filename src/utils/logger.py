import logging
import sys
import os
from datetime import datetime

logger = logging.getLogger("LegalBERT_Project")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = "logs" 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_file_name = os.path.join(log_dir, "run.log")
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO) 
    
    logger.addHandler(file_handler)
    
    logger.propagate = False