import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



processed = Path(r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/data/processed")
print(processed)
datafile = processed / "whatsapp-20240214-112323.parq"
print(datafile)
if not datafile.exists():
    logger.warning("Datafile does not exist. First run src/preprocess.py, and check the timestamp!")

df = pd.read_parquet(datafile)
print(df.dtypes)