# import threading
import re
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from src.settings import Android_Regexes, BaseRegexes, Folders, iOS_Regexes
from src.preprocess import WhatsappPreprocessor
from src.visualisations import DataAnalyzer


def main():

    regexes: BaseRegexes = Android_Regexes()  # type: ignore
    preprocesTime = datetime.now().strftime("%Y%m%d-%H%M%S")

    folders = Folders(
        raw=Path("data/raw"),
        processed=Path("data/processed"),
        datafile=Path("_chat.txt"),
    )
    preprocessor = WhatsappPreprocessor(folders, regexes, preprocesTime)
    preprocessor()

    df = pd.read_parquet(f"data/processed/whatsapp-{preprocesTime}.parq")
    print(df.head())
    
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
    print(df.head())

    # now = datetime.now().strftime("%Y%m%d-%H%M%S")
    # output = processed / f"whatsapp-{now}.csv"

    df.to_csv(f"data/processed/whatsapp-{preprocesTime}.parq")
    df.to_parquet(f"data/processed/whatsapp-{preprocesTime}.parq")


    # Define paths
    datafile_path = f"data/processed/whatsapp-{preprocesTime}.parq"
    print(f"Data file path = {datafile_path}")
    save_location = r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/img/"

    # Create DataAnalyzer instance
    analyzer = DataAnalyzer(datafile_path, save_location)

    # Perform analysis
    analyzer.category_weekday_polar_plot()
    # analyzer.categorie_common_words()
    analyzer.time_wins_mentions(interval='year')
    analyzer.time_bbq_mentions()
    # analyzer.distributie_plot_message_length()
    analyzer.relations_link_vs_media()


if __name__ == "__main__":
    main()


