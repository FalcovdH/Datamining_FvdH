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



processed = Path(r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/data/processed")
print(processed)
datafile = processed / "whatsapp-20240214-112323.parq"
print(datafile)
if not datafile.exists():
    logger.warning("Datafile does not exist. First run src/preprocess.py, and check the timestamp!")

df = pd.read_parquet(datafile)
print(df.dtypes)

# Example usage:
# Assuming 'df' is your DataFrame
fig = create_weekday_line_polar_plot(df)
fig.show()