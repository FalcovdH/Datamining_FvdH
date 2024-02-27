import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



processed = Path("../data/processed")
datafile = processed / "whatsapp-20240213-202158.csv"
if not datafile.exists():
    logger.warning("Datafile does not exist. First run src/preprocess.py, and check the timestamp!")

df = pd.read_csv(datafile)
print(df.dtypes)