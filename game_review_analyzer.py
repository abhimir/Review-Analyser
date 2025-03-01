"""
Mobile Game Review Analysis Tool
--------------------------------
A tool to analyze App Store and Google Play reviews of competing games to gain insights for game development.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st
import json
import time
from datetime import datetime
import re
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import google_play_scraper
from google_play_scraper import app as gplay_app
from google_play_scraper import Sort, reviews as gplay_reviews

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

class MobileGameReviewAnalyzer:
    """Main class for analyzing App Store and Google Play Store reviews"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def fetch_app_id(self, app_name, store='appstore', country='us', language='en'):
        """
        Search for an app in the App Store or Google Play Store and return its ID
        
        Parameters:
        -----------
        app_name : str
            Name of the app to search for
        store : str
            Store to search in ('appstore' or 'googleplay')
        country : str
            Country code for the store (default: 'us')
        language : str
            Language code (default: 'en')
            
        Returns:
        --------
        str
            App ID if found, None otherwise
        """
        if store.lower() == 'appstore':
            search_url = f"https://itunes.apple.com/search?term={app_name.replace(' ', '+')}&entity=software&country={country}"
            response = requests.get(search_url)
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    # Return the first result's ID
                    return str(results[0]['trackId'])
                else:
                    print(f"No apps found with name: {app_name} in App Store")
                    return None
            else:
                print(f"Error searching for app in App Store: {response.status_code}")
                return None
        
        elif store.lower() == 'googleplay':
            try:
                # Try to search for the app
                result = google_play_scraper.search(app_name, lang=language, country=country)
                
                if result and len(result) > 0:
                    # Return the app ID (package name) of the first result
                    return result[0]['appId']
                else:
                    print(f"No apps found with name: {app_name} in Google Play Store")
                    return None
                    
            except Exception as e:
                print(f"Error searching for app in Google Play Store: {e}")
                return None
        
        else:
            print(f"Invalid store type: {store}. Use 'appstore' or 'googleplay'")
            return None
    
    def fetch_reviews(self, app_id, store='appstore', country='us', language='en', max_reviews=5000):
        """
        Fetch reviews for a specific app from the App Store or Google Play Store
        
        Parameters:
        -----------
        app_id : str
            ID of the app to fetch reviews for
        store : str
            Store to fetch reviews from ('appstore' or 'googleplay')
        country : str
            Country code for the store (default: 'us')
        language : str
            Language code (default: 'en')
        max_reviews : int
            Maximum number of reviews to fetch (default: 5000)
            
        Returns:
        --------
        list
            List of review dictionaries
        """
        all_reviews = []
        
        if store.lower() == 'appstore':
            page = 1
            max_pages = 100  # App Store limits to 100 pages max
            
            while page <= max_pages and len(all_reviews) < max_reviews:
                # App Store RSS feed URL for reviews
                url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if we have entries in the response
                    entries = data.get('feed', {}).get('entry', [])
                    
                    # The first entry is metadata, not a review
                    if len(entries) > 1:
                        reviews = entries[1:]  # Skip the first entry
                        
                        # Process and add reviews to our list
                        for review in reviews:
                            try:
                                review_data = {
                                    'id': review.get('id', {}).get('label', ''),
                                    'title': review.get('title', {}).get('label', ''),
                                    'content': review.get('content', {}).get('label', ''),
                                    'rating': int(review.get('im:rating', {}).get('label', 0)),
                                    'version': review.get('im:version', {}).get('label', ''),
                                    'author': review.get('author', {}).get('name', {}).get('label', ''),
                                    'date': review.get('updated', {}).get('label', ''),
                                    'store': 'App Store'
                                }
                                all_reviews.append(review_data)
                            except (KeyError, ValueError) as e:
                                print(f"Error processing App Store review: {e}")
                                continue
                    else:
                        # No more reviews to fetch
                        break
                        
                    page += 1
                    time.sleep(0.5)  # Be nice to the API
                else:
                    print(f"Error fetching App Store reviews: {response.status_code}")
                    break
        
        elif store.lower() == 'googleplay':
            try:
                # Fetch reviews in batches
                # Each continuation token gets us the next batch
                continuation_token = None
                
                while len(all_reviews) < max_reviews:
                    # Google Play Scraper fetches in batches
                    batch_size = min(200, max_reviews - len(all_reviews))
                    if batch_size <= 0:
                        break
                        
                    result, continuation_token = gplay_reviews(
                        app_id,
                        lang=language,
                        country=country,
                        sort=Sort.NEWEST,
                        count=batch_size,
                        continuation_token=continuation_token
                    )
                    
                    if not result:
                        break
                        
                    # Process and add reviews to our list
                    for review in result:
                        try:
                            review_data = {
                                'id': review.get('reviewId', ''),
                                'title': '',  # Google Play reviews don't have titles
                                'content': review.get('content', ''),
                                'rating': review.get('score', 0),
                                'version': '',  # Version info not consistently available
                                'author': review.get('userName', ''),
                                'date': review.get('at', ''),
                                'store': 'Google Play'
                            }
                            all_reviews.append(review_data)
                        except Exception as e:
                            print(f"Error processing Google Play review: {e}")
                            continue
                    
                    # If there's no continuation token, we've reached the end
                    if not continuation_token:
                        break
                        
                    # Status update
                    print(f"Fetched {len(all_reviews)} Google Play reviews so far...")
                    
                    time.sleep(0.5)  # Be nice to the API
                    
            except Exception as e:
                print(f"Error fetching Google Play reviews: {e}")
        
        else:
            print(f"Invalid store type: {store}. Use 'appstore' or 'googleplay'")
            
        print(f"Fetched {len(all_reviews)} reviews from {store}")
        return all_reviews
    
    def create_dataframe(self, reviews):
        """
        Convert review list to a pandas DataFrame
        
        Parameters:
        -----------
        reviews : list
            List of review dictionaries
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing reviews
        """
        df = pd.DataFrame(reviews)
        
        if not df.empty:
            # Convert date string to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Sort by date
            df = df.sort_values('date', ascending=False)
            
        return df
    
    def preprocess_text(self, text):
        """
        Preprocess text for analysis
        
        Parameters:
        -----------
        text : str
            Text to preprocess
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary with sentiment scores
        """
        if not isinstance(text, str) or text.strip() == '':
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0, 'sentiment': 'neutral'}
            
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Add a simple sentiment label
        if scores['compound'] >= 0.05:
            scores['sentiment'] = 'positive'
        elif scores['compound'] <= -0.05:
            scores['sentiment'] = 'negative'
        else:
            scores['sentiment'] = 'neutral'
            
        return scores
    
    def extract_topics(self, reviews_df, n_topics=5, n_top_words=10):
        """
        Extract topics from reviews using LDA
        
        Parameters:
        -----------
        reviews_df : pandas.DataFrame
            DataFrame containing reviews
        n_topics : int
            Number of topics to extract (default: 5)
        n_top_words : int
            Number of top words per topic to return (default: 10)
            
        Returns:
        --------
        tuple
            (topics_df, vectorizer, lda_model)
        """
        # Combine title and content
        reviews_df['processed_text'] = reviews_df['title'].fillna('') + ' ' + reviews_df['content'].fillna('')
        reviews_df['processed_text'] = reviews_df['processed_text'].apply(self.preprocess_text)
        
        # Filter out empty texts
        non_empty_mask = reviews_df['processed_text'].str.strip() != ''
        texts = reviews_df.loc[non_empty_mask, 'processed_text']
        
        if len(texts) == 0:
            print("No valid texts for topic modeling")
            return pd.DataFrame(), None, None
        
        # Create document-term matrix
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
        dtm = vectorizer.fit_transform(texts)
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        lda.fit(dtm)
        
        # Get top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_dict = {
                'topic_id': topic_idx,
                'top_words': top_words,
                'top_words_str': ', '.join(top_words)
            }
            topics.append(topic_dict)
        
        return pd.DataFrame(topics), vectorizer, lda
    
    def identify_key_features(self, df, n_features=20):
        """
        Identify key features mentioned in reviews
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing reviews
        n_features : int
            Number of top features to return (default: 20)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature frequencies
        """
        # Game-specific feature keywords
        feature_keywords = {
            'graphics': ['graphic', 'visual', 'art', 'animation', 'design', 'beautiful'],
            'gameplay': ['gameplay', 'mechanic', 'control', 'play', 'playability'],
            'story': ['story', 'plot', 'narrative', 'character', 'dialogue'],
            'sound': ['sound', 'music', 'audio', 'soundtrack', 'voice', 'sfx'],
            'difficulty': ['difficulty', 'hard', 'easy', 'challenging', 'impossible'],
            'progression': ['level', 'progress', 'upgrade', 'advancement', 'skill tree'],
            'monetization': ['pay', 'purchase', 'microtransaction', 'free', 'coin', 'gem', 'money', 'spend'],
            'bugs': ['bug', 'crash', 'glitch', 'freeze', 'error', 'broken'],
            'multiplayer': ['multiplayer', 'online', 'coop', 'cooperative', 'pvp', 'team'],
            'updates': ['update', 'patch', 'new content', 'expansion', 'dlc'],
            'support': ['support', 'developer', 'customer service', 'response'],
            'ui': ['interface', 'ui', 'menu', 'navigation', 'button', 'control']
        }
        
        # Combine title and content
        df['full_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
        df['full_text'] = df['full_text'].str.lower()
        
        # Count feature mentions
        feature_counts = {feature: 0 for feature in feature_keywords}
        
        for _, row in df.iterrows():
            text = row['full_text']
            for feature, keywords in feature_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        feature_counts[feature] += 1
                        break
        
        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': list(feature_counts.keys()),
            'count': list(feature_counts.values())
        }).sort_values('count', ascending=False)
        
        # Calculate percentage
        total_reviews = len(df)
        feature_df['percentage'] = (feature_df['count'] / total_reviews * 100).round(1)
        
        return feature_df
    
    def analyze_reviews(self, app_name, stores=None, country='us', language='en', max_reviews=2000):
        """
        Main function to analyze reviews for an app from App Store and/or Google Play
        
        Parameters:
        -----------
        app_name : str
            Name of the app to analyze
        stores : list
            List of stores to analyze ('appstore', 'googleplay', or both)
        country : str
            Country code for the stores (default: 'us')
        language : str
            Language code (default: 'en')
        max_reviews : int
            Maximum number of reviews to fetch per store (default: 2000)
            
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        if stores is None:
            stores = ['appstore', 'googleplay']
            
        all_reviews = []
        app_ids = {}
        
        for store in stores:
            # Find app ID
            app_id = self.fetch_app_id(app_name, store, country, language)
            if not app_id:
                print(f"Could not find app: {app_name} in {store}")
                continue
                
            app_ids[store] = app_id
            
            # Fetch reviews
            reviews = self.fetch_reviews(app_id, store, country, language, max_reviews)
            if reviews:
                all_reviews.extend(reviews)
            else:
                print(f"No reviews found for app: {app_name} in {store}")
        
        if not all_reviews:
            return {"error": f"No reviews found for app: {app_name} in any store"}
        
        # Create DataFrame
        reviews_df = self.create_dataframe(all_reviews)
        
        # Analyze sentiment
        reviews_df['sentiment_scores'] = reviews_df['content'].apply(self.analyze_sentiment)
        reviews_df['sentiment'] = reviews_df['sentiment_scores'].apply(lambda x: x['sentiment'])
        reviews_df['compound_score'] = reviews_df['sentiment_scores'].apply(lambda x: x['compound'])
        
        # Extract topics
        topics_df, vectorizer, lda_model = self.extract_topics(reviews_df)
        
        # Identify key features
        features_df = self.identify_key_features(reviews_df)
        
        # Prepare results
        results = {
            "app_name": app_name,
            "app_ids": app_ids,
            "review_count": len(reviews_df),
            "store_distribution": reviews_df['store'].value_counts().to_dict(),
            "average_rating": reviews_df['rating'].mean(),
            "sentiment_distribution": reviews_df['sentiment'].value_counts().to_dict(),
            "reviews_df": reviews_df,
            "topics_df": topics_df,
            "features_df": features_df,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results
        
    def generate_word_cloud(self, text_series, sentiment=None):
        """
        Generate a word cloud from a series of texts
        
        Parameters:
        -----------
        text_series : pandas.Series
            Series of texts to generate word cloud from
        sentiment : str, optional
            Sentiment to filter by (default: None)
            
        Returns:
        --------
        wordcloud.WordCloud
            Generated word cloud
        """
        # Combine all texts
        all_text = ' '.join([self.preprocess_text(text) for text in text_series if isinstance(text, str)])
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(all_text)
        
        return wordcloud
        
# Streamlit UI for the analyzer
def create_app_ui():
    st.set_page_config(
        page_title="Mobile Game Review Analyzer",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Mobile Game Review Analyzer")
    st.write("Analyze App Store and Google Play reviews of competing games to gain insights")
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Analysis Parameters")
        app_name = st.text_input("App Name", "Clash of Clans")
        
        stores = st.multiselect(
            "Stores to Analyze",
            options=["App Store", "Google Play"],
            default=["App Store", "Google Play"]
        )
        
        # Map friendly names to internal names
        store_map = {
            "App Store": "appstore",
            "Google Play": "googleplay"
        }
        
        selected_stores = [store_map[store] for store in stores]
        
        country = st.selectbox("Country", ["us", "gb", "ca", "au", "de", "fr", "jp", "kr", "cn", "br", "ru", "in"], index=0)
        language = st.selectbox("Language", ["en", "fr", "de", "es", "it", "ja", "ko", "pt", "ru", "zh"], index=0)
        page_limit = st.slider("Number of Review Pages/Batches per Store", 1, 20, 5)
        
        analyze_button = st.button("Analyze Reviews")
    
    # Initialize analyzer
    analyzer = MobileGameReviewAnalyzer()
    
    # Display results if available
    if hasattr(st.session_state, 'results'):
        results = st.session_state.results
        
        # App info
        st.header(f"Analysis Results: {results['app_name']}")
        
        # Display App IDs
        col1, col2 = st.columns(2)
        
        with col1:
            for store, app_id in results['app_ids'].items():
                store_name = "App Store" if store == "appstore" else "Google Play"
                st.write(f"{store_name} App ID: {app_id}")
                
        with col2:
            # Display review counts and average rating
            st.write(f"Total reviews analyzed: {results['review_count']}")
            st.write(f"Average rating: {results['average_rating']:.2f}⭐")
            
            # Add export button
            if st.button("Export Analysis to CSV"):
                # Prepare the data for export
                reviews_df = results['reviews_df']
                
                # Convert to CSV and provide a download link
                csv = reviews_df.to_csv(index=False)
                st.download_button(
                    label="Download Reviews CSV",
                    data=csv,
                    file_name=f"{results['app_name']}_reviews.csv",
                    mime="text/csv"
                )
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Sentiment Analysis", 
            "Store Comparison",
            "Feature Breakdown", 
            "Topic Analysis", 
            "Word Cloud",
            "Review Explorer"
        ])
        
        with tab1:
            st.subheader("Sentiment Distribution")
            
            reviews_df = results['reviews_df']
            
            # Sentiment distribution
            sentiment_counts = reviews_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            # Calculate percentages
            total = sentiment_counts['Count'].sum()
            sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total * 100).round(1)
            
            # Create color map
            color_map = {
                'positive': 'green',
                'neutral': 'gray',
                'negative': 'red'
            }
            
            # Create bar chart
            fig = px.bar(
                sentiment_counts, 
                x='Sentiment', 
                y='Count',
                color='Sentiment',
                color_discrete_map=color_map,
                text='Percentage',
                labels={'Count': 'Number of Reviews', 'Sentiment': 'Sentiment'}
            )
            
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(title_text='Review Sentiment Distribution')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating vs Sentiment
            st.subheader("Rating vs Sentiment")
            
            # Create cross-tabulation
            cross_tab = pd.crosstab(
                reviews_df['rating'],
                reviews_df['sentiment'],
                normalize='index'
            ).reset_index().melt(
                id_vars=['rating'],
                var_name='sentiment',
                value_name='percentage'
            )
            
            cross_tab['percentage'] = cross_tab['percentage'] * 100
            
            # Create grouped bar chart
            fig = px.bar(
                cross_tab,
                x='rating',
                y='percentage',
                color='sentiment',
                color_discrete_map=color_map,
                barmode='group',
                labels={
                    'rating': 'Star Rating',
                    'percentage': 'Percentage',
                    'sentiment': 'Sentiment'
                }
            )
            
            fig.update_layout(title_text='Sentiment Distribution by Rating')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment over time
            st.subheader("Sentiment Trend Over Time")
            
            # Group by month and sentiment
            reviews_df['month'] = reviews_df['date'].dt.to_period('M')
            sentiment_time = reviews_df.groupby(['month', 'sentiment']).size().reset_index(name='count')
            sentiment_time['month'] = sentiment_time['month'].astype(str)
            
            # Create line chart
            fig = px.line(
                sentiment_time,
                x='month',
                y='count',
                color='sentiment',
                color_discrete_map=color_map,
                markers=True,
                labels={
                    'month': 'Month',
                    'count': 'Number of Reviews',
                    'sentiment': 'Sentiment'
                }
            )
            
            fig.update_layout(title_text='Sentiment Trend Over Time')
            
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("App Store vs Google Play Comparison")
            
            reviews_df = results['reviews_df']
            
            # Check if we have both stores
            stores = reviews_df['store'].unique()
            
            if len(stores) > 1:
                # Average rating by store
                avg_by_store = reviews_df.groupby('store')['rating'].mean().reset_index()
                
                fig = px.bar(
                    avg_by_store,
                    x='store',
                    y='rating',
                    color='store',
                    color_discrete_map={
                        'App Store': '#0099ff',
                        'Google Play': '#3ddc84'
                    },
                    labels={
                        'store': 'Store',
                        'rating': 'Average Rating'
                    }
                )
                
                fig.update_layout(title_text='Average Rating by Store')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment distribution by store
                sentiment_by_store = pd.crosstab(
                    reviews_df['store'],
                    reviews_df['sentiment'],
                    normalize='index'
                ).reset_index().melt(
                    id_vars=['store'],
                    var_name='sentiment',
                    value_name='percentage'
                )
                
                sentiment_by_store['percentage'] = sentiment_by_store['percentage'] * 100
                
                # Create color map
                color_map = {
                    'positive': 'green',
                    'neutral': 'gray',
                    'negative': 'red'
                }
                
                fig = px.bar(
                    sentiment_by_store,
                    x='store',
                    y='percentage',
                    color='sentiment',
                    color_discrete_map=color_map,
                    barmode='group',
                    labels={
                        'store': 'Store',
                        'percentage': 'Percentage',
                        'sentiment': 'Sentiment'
                    }
                )
                
                fig.update_layout(title_text='Sentiment Distribution by Store')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature mentions by store
                features_list = []
                
                for store_name in stores:
                    store_reviews = reviews_df[reviews_df['store'] == store_name]
                    
                    if len(store_reviews) > 0:
                        # Reuse the feature extraction function
                        store_features = analyzer.identify_key_features(store_reviews)
                        store_features['store'] = store_name
                        features_list.append(store_features)
                
                if features_list:
                    # Combine feature DataFrames
                    combined_features = pd.concat(features_list, ignore_index=True)
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        combined_features,
                        x='feature',
                        y='percentage',
                        color='store',
                        color_discrete_map={
                            'App Store': '#0099ff',
                            'Google Play': '#3ddc84'
                        },
                        barmode='group',
                        labels={
                            'feature': 'Feature',
                            'percentage': 'Percentage Mentioned',
                            'store': 'Store'
                        }
                    )
                    
                    fig.update_layout(title_text='Feature Mentions by Store')
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Comparison not available - reviews from only one store were found.")
                
        with tab3:
            st.subheader("Feature Mentions")
            
            features_df = results['features_df']
            
            # Create horizontal bar chart
            fig = px.bar(
                features_df,
                y='feature',
                x='percentage',
                orientation='h',
                color='percentage',
                color_continuous_scale='Viridis',
                labels={
                    'feature': 'Feature',
                    'percentage': 'Percentage of Reviews Mentioning'
                }
            )
            
            fig.update_layout(title_text='Game Features Mentioned in Reviews')
            fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature sentiment correlation
            st.subheader("Feature Sentiment Analysis")
            
            # Extract features with sentiment
            feature_sentiment = []
            
            for feature in features_df['feature']:
                keywords = feature_keywords = {
                    'graphics': ['graphic', 'visual', 'art', 'animation', 'design', 'beautiful'],
                    'gameplay': ['gameplay', 'mechanic', 'control', 'play', 'playability'],
                    'story': ['story', 'plot', 'narrative', 'character', 'dialogue'],
                    'sound': ['sound', 'music', 'audio', 'soundtrack', 'voice', 'sfx'],
                    'difficulty': ['difficulty', 'hard', 'easy', 'challenging', 'impossible'],
                    'progression': ['level', 'progress', 'upgrade', 'advancement', 'skill tree'],
                    'monetization': ['pay', 'purchase', 'microtransaction', 'free', 'coin', 'gem', 'money', 'spend'],
                    'bugs': ['bug', 'crash', 'glitch', 'freeze', 'error', 'broken'],
                    'multiplayer': ['multiplayer', 'online', 'coop', 'cooperative', 'pvp', 'team'],
                    'updates': ['update', 'patch', 'new content', 'expansion', 'dlc'],
                    'support': ['support', 'developer', 'customer service', 'response'],
                    'ui': ['interface', 'ui', 'menu', 'navigation', 'button', 'control']
                }.get(feature, [])
                
                # Find reviews mentioning this feature
                feature_reviews = reviews_df[reviews_df['full_text'].str.contains('|'.join(keywords), case=False)]
                
                if len(feature_reviews) > 0:
                    # Calculate sentiment distribution
                    sentiment_counts = feature_reviews['sentiment'].value_counts()
                    
                    # Ensure all sentiments are represented
                    for sentiment in ['positive', 'neutral', 'negative']:
                        if sentiment not in sentiment_counts:
                            sentiment_counts[sentiment] = 0
                    
                    # Calculate average compound score
                    avg_score = feature_reviews['compound_score'].mean()
                    
                    feature_sentiment.append({
                        'feature': feature,
                        'positive': sentiment_counts.get('positive', 0),
                        'neutral': sentiment_counts.get('neutral', 0),
                        'negative': sentiment_counts.get('negative', 0),
                        'avg_score': avg_score
                    })
            
            if feature_sentiment:
                feature_sentiment_df = pd.DataFrame(feature_sentiment)
                
                # Calculate total mentions
                feature_sentiment_df['total'] = (
                    feature_sentiment_df['positive'] + 
                    feature_sentiment_df['neutral'] + 
                    feature_sentiment_df['negative']
                )
                
                # Calculate percentages
                for col in ['positive', 'neutral', 'negative']:
                    feature_sentiment_df[f'{col}_pct'] = (
                        feature_sentiment_df[col] / feature_sentiment_df['total'] * 100
                    ).round(1)
                
                # Sort by number of mentions
                feature_sentiment_df = feature_sentiment_df.sort_values('total', ascending=False)
                
                # Create stacked bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=feature_sentiment_df['feature'],
                    x=feature_sentiment_df['positive_pct'],
                    name='Positive',
                    orientation='h',
                    marker=dict(color='green'),
                    text=feature_sentiment_df['positive_pct'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    y=feature_sentiment_df['feature'],
                    x=feature_sentiment_df['neutral_pct'],
                    name='Neutral',
                    orientation='h',
                    marker=dict(color='gray'),
                    text=feature_sentiment_df['neutral_pct'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    y=feature_sentiment_df['feature'],
                    x=feature_sentiment_df['negative_pct'],
                    name='Negative',
                    orientation='h',
                    marker=dict(color='red'),
                    text=feature_sentiment_df['negative_pct'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title_text='Sentiment Distribution by Feature',
                    barmode='stack',
                    xaxis_title='Percentage',
                    yaxis_title='Feature',
                    legend_title='Sentiment'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create heatmap of sentiment scores
                fig = px.bar(
                    feature_sentiment_df,
                    y='feature',
                    x='avg_score',
                    orientation='h',
                    color='avg_score',
                    color_continuous_scale='RdYlGn',
                    labels={
                        'feature': 'Feature',
                        'avg_score': 'Average Sentiment Score'
                    },
                    range_color=[-1, 1]
                )
                
                fig.update_layout(title_text='Feature Sentiment Score (Negative to Positive)')
                
                st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.subheader("Topic Analysis")
            
            topics_df = results['topics_df']
            
            if not topics_df.empty:
                for _, row in topics_df.iterrows():
                    st.write(f"**Topic {row['topic_id'] + 1}:** {row['top_words_str']}")
                
                # Find example reviews for each topic
                st.subheader("Example Reviews by Topic")
                
                # TBD: This would require saving the topic-document matrix
                st.write("Select a topic to see example reviews")
                topic_idx = st.selectbox("Topic", range(1, len(topics_df) + 1), format_func=lambda x: f"Topic {x}")
                
                # Placeholder for topic examples
                st.write("Example reviews would be shown here")
                
            else:
                st.write("No topics were extracted. Try increasing the number of reviews.")
            
        with tab4:
            st.subheader("Word Clouds")
            
            # Generate word clouds
            reviews_df = results['reviews_df']
            
            # Create tabs for different word clouds
            wc_tab1, wc_tab2 = st.tabs(["All Reviews", "By Sentiment"])
            
            with wc_tab1:
                # Generate word cloud for all reviews
                all_wordcloud = analyzer.generate_word_cloud(reviews_df['content'])
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(all_wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                st.pyplot(fig)
                
            with wc_tab2:
                # Create columns for positive and negative word clouds
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Positive Reviews")
                    
                    # Filter positive reviews
                    positive_reviews = reviews_df[reviews_df['sentiment'] == 'positive']
                    
                    if len(positive_reviews) > 0:
                        positive_wordcloud = analyzer.generate_word_cloud(positive_reviews['content'])
                        
                        # Display word cloud
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(positive_wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        
                        st.pyplot(fig)
                    else:
                        st.write("No positive reviews found.")
                
                with col2:
                    st.write("#### Negative Reviews")
                    
                    # Filter negative reviews
                    negative_reviews = reviews_df[reviews_df['sentiment'] == 'negative']
                    
                    if len(negative_reviews) > 0:
                        negative_wordcloud = analyzer.generate_word_cloud(negative_reviews['content'])
                        
                        # Display word cloud
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(negative_wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        
                        st.pyplot(fig)
                    else:
                        st.write("No negative reviews found.")
            
        with tab5:
            st.subheader("Review Explorer")
            
            reviews_df = results['reviews_df']
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rating_filter = st.multiselect(
                    "Filter by Rating",
                    options=[1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5]
                )
                
            with col2:
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment",
                    options=['positive', 'neutral', 'negative'],
                    default=['positive', 'neutral', 'negative']
                )
                
            with col3:
                store_filter = st.multiselect(
                    "Filter by Store",
                    options=reviews_df['store'].unique(),
                    default=reviews_df['store'].unique()
                )
                
            search_term = st.text_input("Search in Reviews", "")
            
            # Apply filters
            filtered_df = reviews_df[
                reviews_df['rating'].isin(rating_filter) &
                reviews_df['sentiment'].isin(sentiment_filter) &
                reviews_df['store'].isin(store_filter)
            ]
            
            if search_term:
                filtered_df = filtered_df[
                    filtered_df['content'].str.contains(search_term, case=False, na=False) |
                    filtered_df['title'].str.contains(search_term, case=False, na=False)
                ]
            
            # Display filters info
            st.write(f"Showing {len(filtered_df)} of {len(reviews_df)} reviews")
            
            # Display reviews
            for idx, row in filtered_df.iterrows():
                # Format the title - Google Play reviews may not have titles
                title = row['title'] if row['title'] else f"Review #{idx}"
                
                # Format date based on type
                if isinstance(row['date'], pd.Timestamp):
                    date_str = row['date'].strftime('%Y-%m-%d')
                else:
                    # Handle string dates from Google Play
                    try:
                        date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                    except:
                        date_str = str(row['date'])
                
                with st.expander(f"{title} ({row['rating']}⭐ - {row['sentiment'].capitalize()} - {row['store']})"):
                    st.write(f"**Date:** {date_str}")
                    
                    if row['version']:
                        st.write(f"**Version:** {row['version']}")
                        
                    st.write(f"**Author:** {row['author']}")
                    st.write(f"**Review:** {row['content']}")
                    st.write(f"**Sentiment Score:** {row['compound_score']:.2f}")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import google_play_scraper
    except ImportError:
        print("Please install the google-play-scraper package with: pip install google-play-scraper")
    
    create_app_ui()
