import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Create a Streamlit app
st.title("Redbus Review Analysis")
st.write("Upload a CSV file to perform sentiment analysis")

# Create a file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

def read_csv_file(file):
    """Read a CSV file into a Pandas dataframe"""
    try:
        return pd.read_csv(file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.error(f"Error reading CSV file: {e}")
        return None

def perform_sentiment_analysis(df, review_column):
    """Perform sentiment analysis on a column of a Pandas dataframe"""
    sentiments = []
    for text in df[review_column]:
        sentiment = sentiment_analyzer.polarity_scores(text)
        sentiments.append(sentiment)
    return pd.DataFrame(sentiments)

def create_charts(sentiment_df):
    """Create pie and bar charts of the average sentiment scores"""
    avg_sentiments = sentiment_df.drop("compound", axis=1).mean()
    fig_pie = px.pie(values=avg_sentiments.values, names=avg_sentiments.index, title="Average Sentiment Scores (Pie Chart)")
    st.plotly_chart(fig_pie)

def display_sentiment_scores(sentiment_df):
    """Display the sentiment scores and overall sentiment"""
    st.write("Sentiment Scores:")
    st.write(sentiment_df)
    overall_sentiment = sentiment_df["compound"].mean()
    st.write(f"Overall Sentiment: {overall_sentiment:.2f}")

def display_summary(df, review_column, sentiment_df):
    """Display a summary of the reviews"""
    overall_sentiment = sentiment_df["compound"].mean()
    if overall_sentiment > 0:
        conclusion = "positive"
    elif overall_sentiment < 0:
        conclusion = "negative"
    else:
        conclusion = "neutral"
    st.write(f"Conclusion: The overall sentiment of the reviews is {conclusion}.")
    negative_reviews = df[df[review_column].apply(lambda x: sentiment_analyzer.polarity_scores(x)["compound"] < 0)]
    st.write("Negative Reviews:")
    st.write(negative_reviews[review_column])

if uploaded_file is not None:
    df = read_csv_file(uploaded_file)
    if df is not None:
        review_column = st.selectbox("Select a column to analyze", df.columns)
        sentiment_df = perform_sentiment_analysis(df, review_column)
        create_charts(sentiment_df)
        display_summary(df, review_column, sentiment_df)
        display_sentiment_scores(sentiment_df)

        # Display a sentiment chart
        st.write("Sentiment Chart:")
        st.bar_chart(sentiment_df["compound"])

