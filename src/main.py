import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import re
import regex
import numpy as np
import emoji
import os
import plotly.express as px
import plotly.io as pio
import plotly.subplots as sp
import plotly.graph_objects as go
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import datetime
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator




# # Assuming 'df' is your DataFrame with 'timestamp' and 'message' columns
# df['weekday'] = df['timestamp'].dt.day_name()
# df['year'] = df['timestamp'].dt.year

# weekday_grouped_msg = (df.groupby(['year', 'weekday'])['message']
#                           .value_counts()
#                           .groupby(['year', 'weekday'])
#                           .sum()
#                           .reset_index(name='count'))

# # Sort the 'weekday' column in the desired order
# weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# weekday_grouped_msg['weekday'] = pd.Categorical(weekday_grouped_msg['weekday'], categories=weekday_order, ordered=True)

# # Sort the DataFrame by 'year' and 'weekday'
# weekday_grouped_msg = weekday_grouped_msg.sort_values(by=['year', 'weekday'])
# category_orders = {'weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

# colors = np.where(weekday_grouped_msg['year'] < 2021, 'blue', 'red')

# fig = px.line_polar(weekday_grouped_msg, r='count', theta='weekday', color=colors,
#                     line_close=True, template='plotly_dark', category_orders=category_orders)

# max_count_value = weekday_grouped_msg['count'].max()
# fig.update_traces(fill='toself')
# fig.update_layout(
#     polar=dict(radialaxis=dict(visible=True, title='Aantal berichten')),
#     showlegend=True,
#     title='Aantal berichten per dag en jaren van een zaalvoetbal team'
# )

# annotation_text = "Vanaf 2022 is de speeldag van voetbal verplaatst van woensdag naar vrijdag. <br> Dit is terug te zien in het aantal berichten per dag"
# fig.add_annotation(
#     go.layout.Annotation(
#         text=annotation_text,
#         xref="paper", yref="paper",
#         x=0.02, y=.950,
#         showarrow=False,
#         font=dict(size=12, color="white"),
#     )
# )

# # img_directory = 'img'
# # os.makedirs(img_directory, exist_ok=True)

# # Save the plot as an HTML file
# # png_file_path = os.path.join(r'C:/Users/a427617\Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024', img_directory, 'msg_count_weekday_year.jpeg')

# # fig.write_image(png_file_path, format='jpeg')  


# # print(f"Plot saved as HTML: {png_file_path}")

# fig.show()

def create_weekday_line_polar_plot(df):
    # Extract weekday and year from timestamp
    df['weekday'] = df['timestamp'].dt.day_name()
    df['year'] = df['timestamp'].dt.year

    # Group by year, weekday, and message, and calculate the count
    weekday_grouped_msg = (df.groupby(['year', 'weekday'])['message']
                           .value_counts()
                           .groupby(['year', 'weekday'])
                           .sum()
                           .reset_index(name='count'))

    # Sort the 'weekday' column in the desired order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_grouped_msg['weekday'] = pd.Categorical(weekday_grouped_msg['weekday'], categories=weekday_order, ordered=True)

    # Sort the DataFrame by 'year' and 'weekday'
    weekday_grouped_msg = weekday_grouped_msg.sort_values(by=['year', 'weekday'])
    category_orders = {'weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

    # Define colors based on the year
    colors = np.where(weekday_grouped_msg['year'] < 2021, 'blue', 'red')

    # Create a polar line plot using Plotly Express
    fig = px.line_polar(weekday_grouped_msg, r='count', theta='weekday', color=colors,
                        line_close=True, template='plotly_dark', category_orders=category_orders)

    # Customize the figure
    max_count_value = weekday_grouped_msg['count'].max()
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, title='Aantal berichten')),
        showlegend=True,
        title='Aantal berichten per dag en jaren van een zaalvoetbal team'
    )

    # Add annotation to the figure
    annotation_text = "Vanaf 2022 is de speeldag van voetbal verplaatst van woensdag naar vrijdag. <br> Dit is terug te zien in het aantal berichten per dag"
    fig.add_annotation(
        go.layout.Annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=.950,
            showarrow=False,
            font=dict(size=12, color="white"),
        )
    )

    return fig

def create_mentions_over_time_plot(df, interval='year'):

    # Convert the 'timestamp' column to datetime, if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract month, year, or both from timestamp based on the specified interval
    if interval == 'month':
        df['time_interval'] = df['timestamp'].dt.to_period('M').astype(str)
    elif interval == 'year':
        df['time_interval'] = df['timestamp'].dt.to_period('Y').astype(str)
    else:
        raise ValueError("Invalid interval. Supported values are 'month' and 'year'.")

    # Exclude data for the last year (2024)
    df = df[df['timestamp'].dt.year < 2024]

    # Define regex patterns for each word category
    word_patterns = {
        'Gewonnen': re.compile(r'\b(?:Win|winst|gewonnen)\b', re.IGNORECASE),
        'Verloren': re.compile(r'\b(?:Lose|verloren)\b', re.IGNORECASE),
        'Gelijk': re.compile(r'\b(?:Gelijk|gelijkspel)\b', re.IGNORECASE)
    }
    
    # Count occurrences of each word category in each message
    for category, pattern in word_patterns.items():
        df[f'{category}_mentions'] = df['message'].apply(lambda x: len(pattern.findall(x)))

    # Group by time_interval and calculate the total number of mentions for each word category
    mentions_over_time = df.groupby('time_interval')[[f'{category}_mentions' for category in word_patterns]].sum().reset_index()

    # Set specific line colors for each category
    color_palette = {'Gewonnen': 'green', 'Verloren': 'red', 'Gelijk': 'blue'}

    # Set the seaborn style
    sns.set(style="darkgrid")

    # Create a line plot using Seaborn
    plt.figure(figsize=(10, 6))
    for category in word_patterns.keys():
        sns.lineplot(data=mentions_over_time, x='time_interval', y=f'{category}_mentions', label=category, color=color_palette[category])

    # Customize the plot
    plt.title(f'Winst/verlies vermeldingen (Gegroepeerd bij {interval.capitalize()})')
    plt.xlabel('Jaren')
    plt.ylabel('Winst/Verlies')
    plt.legend()

    # Set x-axis ticks to every 1 unit of the specified interval
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.gca().set_xticks(mentions_over_time['time_interval'])

    plt.annotate('Tijd voor een transfer', xy=(0.80, 0.95), xytext=(0.65, 0.95),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1),
                 fontsize=11, ha='center', va='center')
    
    plt.savefig(r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/img/Gewonnen_verloren.png")
    plt.show()

def create_bbq_mentions_over_time_plot(df, save_path=None):
    # Convert the 'timestamp' column to datetime, if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract month from timestamp
    df['month'] = df['timestamp'].dt.month

    # Define a regex pattern to search for mentions of "BBQ" or "bbq" in the 'message' column
    bbq_pattern = re.compile(r'\b(?:BBQ|Bbq|bbq|barbecue|barbeque|bbq\'en|bbq\'ing)\b', re.IGNORECASE)

    # Count occurrences of "BBQ" or "bbq" in each message
    df['bbq_mentions'] = df['message'].apply(lambda x: len(bbq_pattern.findall(x)))

    # Group by month and calculate the average number of BBQ mentions
    bbq_mentions_over_time = df.groupby('month')['bbq_mentions'].mean().reset_index()

    # Set the seaborn style
    sns.set(style="darkgrid")

    # Create a line plot using Seaborn
    plt.figure(figsize=(20, 5))
    sns.lineplot(data=bbq_mentions_over_time, x='month', y='bbq_mentions', marker='o', color='blue')

    # Customize the plot
    plt.title('Gemiddelde aantal vermeldingen BBQ in de tekst (Grouped by Month)')
    plt.xlabel('Maand')
    plt.ylabel('Gemiddeld aantal vermeldingen BBQ')

    plt.annotate('Wanneer heeft de BBQ plaatsgevonden?? \n \n Antwoord: Nog nooit!', xy=(6, 4.5), xytext=(0.35, 0.9),
                 xycoords='axes fraction', textcoords='axes fraction',
                 fontsize=11, ha='center', va='center')
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

processed = Path(r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/data/processed")
print(processed)
datafile = processed / "whatsapp-20240214-112323.parq"
print(datafile)
if not datafile.exists():
    logger.warning("Datafile does not exist. First run src/preprocess.py, and check the timestamp!")

df = pd.read_parquet(datafile)
# print(df.dtypes)


fig = create_weekday_line_polar_plot(df)
fig.show()
create_mentions_over_time_plot(df, interval='year')
create_bbq_mentions_over_time_plot(df, save_path=r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/img/bbq_vermeldingen.png")