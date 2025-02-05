import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#import sys
#sys.path.append('c:/users/pratik bhosale\appdata\local\programs\python\python313\lib\site-packages (2024.10.0)')

from sklearn.metrics import mean_squared_error
import seaborn as sns


import io

st.set_page_config(layout="wide")

# Set up Streamlit page
st.title("Zomato Data Analysis and Restaurant Recommendation App")

# File path for the default dataset
DEFAULT_FILE_PATH = "zomato.csv"

# Load default dataset
def load_default_data():
    try:
        data = pd.read_csv(DEFAULT_FILE_PATH, encoding='ISO-8859-1')
        return data
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        return None

# Load the data
data = load_default_data()

# Page Navigation (Centered)
page = st.radio("Select a page:", ("Home", "Data Analysis", "Recommendation System", "Predictive Modeling"), index=0, horizontal=True)

# Home Page
if page == "Home":
    st.subheader("Welcome to the Zomato Data Analysis and Restaurant Recommendation App")
    st.write("""
    This app allows you to explore and analyze restaurant data from Zomato, visualize trends, and generate restaurant recommendations based on user preferences.
    
    **Features:**
    - View dataset and its summary information
    - Explore the distribution of ratings and top cuisines
    - Filter data interactively by city and price range
    - Get restaurant recommendations based on cuisine similarity
    - Predict restaurant ratings using machine learning

    **How to use:**
    1. Use the top navigation to select between different sections of the app.
    2. In the 'Data Analysis' section, view and analyze data such as ratings and cuisines.
    3. In the 'Recommendation System', get restaurant suggestions based on cuisine similarity.
    4. Use the 'Predictive Modeling' section to predict aggregate ratings using a regression model.

    The dataset is loaded from a default path, and you can download filtered data at any time.
    """)

# Data Analysis Section
elif page == "Data Analysis":
    if data is not None:
        show_dataset = st.checkbox("Show Dataset")
        show_info = st.checkbox("Show Dataset Info")
        check_missing_values = st.checkbox("Check Missing Values")
        show_rating_distribution = st.checkbox("Distribution of Aggregate Ratings")
        show_top_cuisines = st.checkbox("Top 10 Cuisines")
        interactive_filtering = st.checkbox("Interactive Data Filtering")
        show_city_distribution = st.checkbox("Show City-Wise Distribution")

        # Display dataset
        if show_dataset:
            st.write(data.head())

        # General Information
        if show_info:
            buffer = io.StringIO()  # Create an in-memory text stream
            data.info(buf=buffer)  # Capture dataset info in buffer
            content = buffer.getvalue()  # Get buffer content
            st.text(content)  # Display the content in Streamlit

        if check_missing_values:
            st.write(data.isnull().sum())

        # Drop rows with missing Cuisines
        data = data.dropna(subset=['Cuisines'])

        # Data Insights
        st.subheader("Data Insights")
        if show_rating_distribution:
            fig, ax = plt.subplots()
            sns.histplot(data['Aggregate rating'], kde=True, bins=20, ax=ax)
            ax.set_title('Distribution of Aggregate Ratings')
            st.pyplot(fig)

        if show_top_cuisines:
            cuisine_counts = data['Cuisines'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, ax=ax)
            ax.set_title('Top 10 Cuisines')
            st.pyplot(fig)

        # Interactive Data Filtering
        if interactive_filtering:
            st.subheader("Interactive Data Filtering")
            cities = st.multiselect("Select Cities", options=data['City'].unique(), default=None)
            price_range = st.multiselect("Select Price Range", options=data['Price range'].unique(), default=None)

            filtered_data = data
            if cities:
                filtered_data = filtered_data[filtered_data['City'].isin(cities)]
            if price_range:
                filtered_data = filtered_data[filtered_data['Price range'].isin(price_range)]

            st.write(f"Filtered Data: {filtered_data.shape[0]} rows")
            st.dataframe(filtered_data)

            # Option to download filtered data
            st.subheader("Download Processed Data")
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(filtered_data)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name='filtered_zomato_data.csv',
                mime='text/csv',
            )

        # City-wise distribution of restaurants
        if show_city_distribution:
            st.subheader("City-Wise Distribution of Restaurants")
            city_counts = data['City'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=city_counts.values, y=city_counts.index, ax=ax)
            ax.set_title('Top Cities with Most Restaurants')
            st.pyplot(fig)
    else:
        st.warning("Data not found.")

# Recommendation System
elif page == "Recommendation System":
    if data is not None:
        st.subheader("Restaurant Recommendation System")
        vectorizer = CountVectorizer()
        cuisine_matrix = vectorizer.fit_transform(data['Cuisines'].fillna(''))
        similarity = cosine_similarity(cuisine_matrix)

        def recommend_restaurants(restaurant_name, n=5):
            if restaurant_name not in data['Restaurant Name'].values:
                return "Restaurant not found."
            idx = data[data['Restaurant Name'] == restaurant_name].index[0]
            scores = list(enumerate(similarity[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            recommended_idx = [i[0] for i in scores[1:n+1]]
            return data.iloc[recommended_idx][['Restaurant Name', 'Cuisines', 'Aggregate rating']]

        restaurant_name = st.text_input("Enter a Restaurant Name for Recommendations")
        n_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
        if st.button("Get Recommendations"):
            if restaurant_name:
                recommendations = recommend_restaurants(restaurant_name, n_recommendations)
                st.write(recommendations)
    else:
        st.warning("Data not found.")

# Predictive Modeling
elif page == "Predictive Modeling":
    if data is not None:
        st.subheader("Predictive Modeling: Aggregate Rating Prediction")
        features = data[['Average Cost for two', 'Votes', 'Price range']]
        target = data['Aggregate rating']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")

        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=features.columns, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)

        # Additional evaluation metrics for predictive modeling
        st.subheader("Model Evaluation Metrics")
        r2_score = model.score(X_test, y_test)
        st.write(f"R-squared: {r2_score:.2f}")
    else:
        st.warning("Data not found.")
