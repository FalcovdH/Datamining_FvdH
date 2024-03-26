from pathlib import Path
from loguru import logger
import pandas as pd
from datetime import datetime

processed = Path("../data/processed")
datafile = processed / "whatsapp-20240325-161753.csv"
if not datafile.exists():
    logger.warning("Datafile does not exist. First run src/preprocess.py, and check the timestamp!")

df = pd.read_csv(datafile, parse_dates=["timestamp"])
df.head()

import re

emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"  # Dingbats
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)

def has_emoji(text):
    return bool(emoji_pattern.search(text))

df['has_emoji'] = df['message'].apply(has_emoji)

import re
clean_tilde = r"^~\u202f"
df["author"] = df["author"].apply(lambda x: re.sub(clean_tilde, "", x))

has_link = r"http"
df['has_link'] = df['message'].str.contains(has_link)

df = df.drop(index=[0])

now = datetime.now().strftime("%Y%m%d-%H%M%S")
output = processed / f"whatsapp-{now}.csv"

df.to_csv(output, index=False)
df.to_parquet(output.with_suffix(".parq"), index=False)