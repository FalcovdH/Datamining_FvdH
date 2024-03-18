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
from matplotlib import ticker
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


def plot_message_length(df):
    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    
    # Exclude the years 2017 and 2024
    df_filtered = df[(df['year'] != 2017) & (df['year'] != 2024)].copy()  # Use copy() to create a copy

    # Extract quarter from timestamp
    df_filtered['quarter'] = df_filtered['timestamp'].dt.quarter

    # Specify the save location
    save_location = r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/img/"

    # Create a FacetGrid for message_length per day of the week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    g_day = sns.FacetGrid(df_filtered, col='day_of_week', col_wrap=4, height=3, col_order=days_order)
    g_day.map(sns.histplot, 'message_length', color='green', kde=True)  # Use 'message_length' instead of x="message_length"
    g_day.set(xticks=range(0, 300, 50), xlabel="Message Length", ylabel="Count")  # Set x-axis values and labels
    g_day.despine(left=True, bottom=False)
    g_day.fig.suptitle('Message Length per Day of the Week', y=1.02)

    # Create a FacetGrid for message_length per quarter of the year
    quarters_order = [1, 2, 3, 4]
    g_quarter = sns.FacetGrid(df_filtered, col='quarter', col_wrap=4, height=3, col_order=quarters_order)
    g_quarter.map(sns.histplot, 'message_length', color='blue', kde=True)  # Use 'message_length' instead of x="message_length"
    g_quarter.set(xticks=range(0, 300, 50), xlabel="Message Length", ylabel="Count")  # Set x-axis values and labels
    g_quarter.despine(left=True, bottom=False)
    g_quarter.fig.suptitle('Message Length per Quarter of the Year', y=1.02)

    # Create a FacetGrid for message_length per year
    g_year = sns.FacetGrid(df_filtered, col='year', col_wrap=4, height=3)
    g_year.map(sns.histplot, 'message_length', color='orange', kde=True)  # Use 'message_length' instead of x="message_length"
    g_year.set(xticks=range(0, 300, 50), xlabel="Message Length", ylabel="Count")  # Set x-axis values and labels
    g_year.despine(left=True, bottom=False)
    g_year.fig.suptitle('Message Length per Year', y=1.02)

    # Set x-axis limit
    for ax in g_day.axes.flatten():
        ax.set_xlim(0, 250)

    for ax in g_quarter.axes.flatten():
        ax.set_xlim(0, 250)

    for ax in g_year.axes.flatten():
        ax.set_xlim(0, 250)

    g_day.savefig(save_location + 'distributie_message_length_per_day_of_week.png')  # Save the plot
    g_quarter.savefig(save_location + 'distributie_message_length_per_quarter.png')  # Save the plot
    g_year.savefig(save_location + 'distributie_message_length_per_year.png')  # Save the plot
    # Show the plots
    plt.show()


def plot_mean_message_vs_media(df):
    """
    Plot mean message length vs. mean media presence for authors in DataFrame df.
    Highlight team leads in red with bold font.
    """
    # Add a new column 'has_media' indicating whether "<Media weggelaten>" is present
    df['has_media'] = df['message'].str.contains(r'<Media weggelaten>', case=False, regex=True)

    # Create a column to identify team members
    team_members = [
        'Belly Zaalvoetbal', 'Bryan Zaagsma', 'Casper Guit', 'Falco',
        'Jeroen Huter', 'Jeroen Zaaltvoetbal', 'Kay Jacobs', 'Kevin Zaagsma',
        'Romano Mundo', 'Ruben Zaalvoetbal', 'Tom Danko'
    ]
    df['is_team_member'] = df['author'].isin(team_members)

    # Group by "author" and aggregate the necessary columns
    p = df.groupby(["author"]).agg({
        "message_length": "mean",
        "has_media": "sum",
        "author": "count",
        "is_team_member": "first",  # Take the first value since it's the same for each author
    }).rename(columns={"author": "count"})

    # Filter rows with count greater than 10
    p = p[p["count"] > 10]

    # Create scatterplot
    ax = sns.scatterplot(data=p, x="message_length", y="has_media", size="count", sizes=(10, 500), alpha=0.3, hue="is_team_member", palette={False: 'gray', True: 'red'})

    # Add title and subtitle with smaller font size
    plt.title("Comparison of teammembers (red) and ex-teammembers (gray)", fontsize=9)
    plt.suptitle("Mean Message Length vs. Mean Media Presence", fontsize=12)

    # Show names of the top 3 authors with the highest sum of has_media
    top_3_has_media = p.nlargest(3, 'has_media')['has_media']
    for i, (author, sum_has_media) in enumerate(top_3_has_media.items()):
        plt.text(p.loc[author, "message_length"], p.loc[author, "has_media"], f"{author}: {sum_has_media}", fontsize=8, fontweight='bold')

    # Customizing legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], title="Team Member", fontsize=8, title_fontsize=9)

    # Set a formatter for the legend to use a decimal separator
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    plt.savefig(r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/img/relation_message_media.png")
    plt.show()


processed = Path(r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/data/processed")
print(processed)
datafile = processed / "whatsapp-20240214-112323.parq"
print(datafile)
if not datafile.exists():
    logger.warning("Datafile does not exist. First run src/preprocess.py, and check the timestamp!")

df = pd.read_parquet(datafile)
# print(df.dtypes)


# fig = create_weekday_line_polar_plot(df)
# fig.show()
# create_mentions_over_time_plot(df, interval='year')
# create_bbq_mentions_over_time_plot(df, save_path=r"C:/Users/a427617/Documents/Master Data science/Blok 3 - Data mining/Data-Mining---2024/img/bbq_vermeldingen.png")
plot_message_length(df)
plot_mean_message_vs_media(df)