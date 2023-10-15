import os
from pathlib import Path
import sys
import logging

logging_format = "[%(asctime)s: %(levelname)s: %(module)s] - %(message)s"
#log_dir = "./auto_learn/logs"
#log_filepath = f"./auto_learn/{log_dir}/'auto_learn_logs.log'"
#os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format=logging_format, 
    handlers=[
        #logging.FileHandler(log_filepath), 
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Logger")
