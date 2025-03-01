"""
Mobile Game Review Analysis Tool
--------------------------------
A tool to analyze App Store and Google Play reviews of competing games to gain insights for game development.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import re
import json
import time
from datetime import datetime
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for cloud environments
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Custom NLTK setup for cloud environments
import nltk

# Create a directory for NLTK data in the current working directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data to the custom directory
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('vader_lexicon', download_dir=nltk_data_dir)
    
    # Import NLTK components after download
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception as e:
    st.error(f"Error setting up NLTK: {e}")
    st.info("Try refreshing the page. If the error persists, contact the administrator.")

# Import scikit-learn components
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import WordCloud with error handling
try:
    from wordcloud import WordCloud
except ImportError:
    st.warning("WordCloud package not available - word cloud visualization will be disabled.")
    class WordCloud:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            return None

# Import Google Play Scraper
try:
    import google_play_scraper
    from google_play_scraper import app as gplay_app
    from google_play_scraper import Sort, reviews as gplay_reviews
except ImportError:
    st.error("Google Play Scraper not available - Google Play analysis will be disabled.")
    st.info("Contact the administrator to install the google-play-scraper package.")


class MobileGameReviewAnalyzer:
    """Main class for analyzing App Store and Google Play Store reviews"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            st.error(f"Error initializing natural language processing components: {e}")
            st.info("Some analysis features may not work correctly.")
            self.stop_words = set()
            self.lemmatizer = None
            self.sentiment_analyzer = None
    
    def extract_app_id_from_url(self, url):
        """
        Extract app ID from App Store or Google Play URL
        
        Parameters:
        -----------
        url : str
            URL of the app
            
        Returns:
        --------
        tuple
            (store_type, app_id) if successful, (None, None) otherwise
        """
        if not url:
            return None, None
            
        # Check if it's an App Store URL
        app_store_pattern = r'apps\.apple\.com/.*?/app/.*?/id(\d+)'
        app_store_match = re.search(app_store_pattern, url)
        
        if app_store_match:
            app_id = app_store_match.group(1)
            return 'appstore', app_id
            
        # Check if it's a Google Play URL
        google_play_pattern = r'play\.google\.com/store/apps/details\?id=([^&]+)'
        google_play_match = re.search(google_play_pattern, url)
        
        if google_play_match:
            app_id = google_play_match.group(1)
            return 'googleplay', app_id
            
        return None, None
