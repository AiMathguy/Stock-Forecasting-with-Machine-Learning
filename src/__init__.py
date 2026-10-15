import sys
import os

import logging

logging_str="%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=logging_str)

log_dir=os.path.join(os.getcwd(),"logs")
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(os.path.join(log_dir,"running_logs.log"))])


