# 🚌 Redbus Review Sentiment Analyzer

A user-friendly web application built with Streamlit and Python for performing sentiment analysis on customer reviews from a CSV file. This tool is specifically designed to analyze large datasets of text (such as Redbus customer feedback) to determine overall sentiment, visualize positive/negative trends, and pinpoint specific areas of concern.

## ✨ Features

  * **CSV Upload:** Easily upload any CSV file containing customer feedback or text data.
  * **Column Selection:** Select the exact column in your CSV that contains the text data (e.g., "review\_text").
  * **VADER Sentiment Analysis:** Utilizes the robust **VADER** (Valence Aware Dictionary and sEntiment Reasoner) model from `nltk` for accurate social media and customer review sentiment scoring.
  * **Visual Conclusion:** Provides a clear conclusion (Positive, Negative, or Neutral) based on the overall compound score.
  * **Interactive Charts:** Displays a Plotly Pie Chart to visualize the average distribution of positive, negative, and neutral scores.
  * **Negative Review Highlight:** Explicitly lists the reviews that contributed to a negative sentiment for quick action or investigation.

## 🛠️ Technology Stack

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **App Framework** | `streamlit` | Creating the interactive web interface. |
| **Data Handling** | `pandas` | Reading, manipulating, and structuring CSV data. |
| **Sentiment Analysis** | `nltk` (VADER) | Core sentiment scoring engine. |
| **Visualization** | `plotly.express` | Generating interactive and clear data visualizations (Pie Chart, Bar Chart). |

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

You need Python 3.x installed.

1.  **Install Required Libraries:**
    Install all necessary Python packages:

    ```bash
    pip install streamlit pandas nltk plotly
    ```

2.  **Download NLTK Data:**
    The VADER model needs to be downloaded the first time `nltk` is used.

    *Open a Python interpreter or run the code below in a script:*

    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

### Running the App

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/dhruv-dosh/Sentiment_Analysis_On_Bus_Reviews](https://github.com/dhruv-dosh/Sentiment_Analysis_On_Bus_Reviews)
    cd Sentiment_Analysis_On_Bus_Reviews
    ```

2.  **Execute the Streamlit Application:**
    If your script is named `app.py`:

    ```bash
    streamlit run app.py
    ```

3.  **Access:** The application will automatically open in your default web browser at `http://localhost:8501`.

## 📝 How to Use

1.  **Upload:** Click **"Choose a CSV file"** and upload your review data.
2.  **Select Column:** Use the dropdown menu **"Select a column to analyze"** to choose the text column containing the reviews (e.g., `Review_Text`, `Comment`).
3.  **Review Results:** The app will instantly display the charts, overall sentiment score, conclusion, and a list of negative reviews.

-----

**Author:** \ Dhruv Doshi.
