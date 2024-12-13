"""
This script is a Streamlit-based dashboard for smart energy analysis. 
It integrates advanced data processing, visualization, and AI-powered insights to analyze 
energy usage patterns, identify trends, and provide actionable insights.

Modules Used:
- streamlit: For building the interactive web app.
- pandas: For data manipulation and processing.
- plotly: For creating interactive visualizations.
- prophet: For time series forecasting.
- google.generativeai: For AI-driven insights using Google's Generative AI.
- langchain: For integrating and chaining language models.
"""

# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from openmeteopy import OpenMeteo
from openmeteopy.hourly import HourlyForecast
from openmeteopy.options import ForecastOptions
from openmeteopy.hourly import HourlyHistorical
from openmeteopy.daily import DailyHistorical
from openmeteopy.options import HistoricalOptions
from openmeteopy.utils.constants import *
import os
import holidays
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score
import calendar
import numpy as np

# Configure the Google Generative AI API
GOOGLE_API_KEY = "AIzaSyCppzDCO2eepEd3qLF7oWpI3Nkz4bW-TGc"
genai.configure(api_key=GOOGLE_API_KEY)

def load_custom_css(css_file_path):
    try:
        with open(css_file_path, "r") as css_file:
            custom_styles = css_file.read()
            return custom_styles
    except FileNotFoundError:
        st.error(f"CSS file not found at: {css_file_path}")
		
# Class for loading and processing data
model_path = 'D:/IE Master/Third Term/WattSolution-main\WattSolution-main/models/energy_consumption/Multi_city'
class DataLoader:
    """
    Handles loading and processing of energy data.

    Methods:
    - load_and_process_data: Reads energy data from a CSV file, processes it, and returns a clean DataFrame.
    """
    
    @staticmethod
    def load_and_process_data():
        """"Load energy data from a CSV file, perform preprocessing, and return a cleaned DataFrame."""
        try:
            df = pd.read_csv("energy_data.csv") # Load data from the file
            
            # Extract and process datetime features
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['hour'].astype(float)
            df['day_of_week'] = df['date'].dt.day_name()
            df['week'] = df['date'].dt.isocalendar().week
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.strftime('%B')
            df['year'] = df['date'].dt.year
            
            # Monthly aggregation
            df['monthly_usage'] = df.groupby(['year', 'month'])['usage'].transform('sum')
            df['monthly_avg'] = df.groupby(['year', 'month'])['usage'].transform('mean')
            
            # Add categorical and derived features
            df['quarter'] = df['date'].dt.quarter
            df['day_type'] = df.apply(
                lambda x: 'Holiday' if x['is_holiday'] else 
                ('Weekend' if x['is_weekend'] else 'Weekday'), axis=1
            )
            
            # Categorize temperatures into quantiles
            df['temp_category'] = pd.qcut(
                df['temp'], 
                q=5, 
                labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']
            )
            
            # Return the cleaned data
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

# Class for filtering data
class DataFilter:
    """
    Provides filtering functionality for energy data.

    Methods:
    - get_city_date_range: Determines the date range for the selected cities.
    - apply_filters: Applies selected filters such as date range, cities, and appliances to the data.
    """
    @staticmethod
    def get_city_date_range(df, selected_cities):
        """Return the minimum and maximum dates for the selected cities."""
        try:
            filtered_df = df[df['city'].isin(selected_cities)]
            min_date = filtered_df['date'].min().date()
            max_date = filtered_df['date'].max().date()
            return min_date, max_date
        except Exception as e:
            st.error(f"Error determining date range: {str(e)}")
            return df['date'].min().date(), df['date'].max().date()

    @staticmethod
    def apply_filters(df, start_date, end_date, cities, appliances):
        """Filter data based on user-selected date range, cities, and appliances."""
        try:
            mask = (
                (df['date'].dt.date >= start_date) &
                (df['date'].dt.date <= end_date) &
                (df['city'].isin(cities)) &
                (df['appliance'].isin(appliances))
            )
            return df[mask]
        except Exception as e:
            st.error(f"Error applying filters: {str(e)}")
            return df


class InsightGenerator:
    """
    Generates insights and provides AI-driven responses based on the filtered data.

    Methods:
    - generate_context: Creates a detailed context for AI analysis based on the filtered data.
    - initialize_qa_chain: Initializes the AI-powered question-answering chain.
    - get_analysis: Generates a response to user queries based on the AI analysis.
    """
    
    def __init__(self, df):
        self.df = df

    def generate_context(self, filtered_df):
        """Generate detailed textual context from the filtered data."""
        context = []

        # Overall comparison context for all cities and appliances
        grid_usage = (
            filtered_df[filtered_df['appliance'] == 'grid']
            .groupby(['year', 'month_name'])['usage']
            .sum()
            .reset_index()
            .sort_values(['year', 'month_name'])
        )
        if not grid_usage.empty:
            context.append(f"Overall Grid Usage Comparison per Month:\n{grid_usage.to_dict('records')}")

        for city in filtered_df['city'].unique():
            city_data = filtered_df[filtered_df['city'] == city]

            for appliance in city_data['appliance'].unique():
                app_data = city_data[city_data['appliance'] == appliance]

                # Monthly analysis
                monthly_usage = app_data.groupby(['year', 'month_name'])['usage'].agg([
                    ('total_usage', 'sum'),
                    ('average_usage', 'mean')
                ]).round(3)

                # Sort months chronologically
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                            'July', 'August', 'September', 'October', 'November', 'December']
                monthly_usage = monthly_usage.reset_index()
                monthly_usage['month_order'] = monthly_usage['month_name'].map(
                    {month: idx for idx, month in enumerate(month_order)}
                )
                monthly_usage = monthly_usage.sort_values(['year', 'month_order'])

                # Peak months and yearly trends
                if not monthly_usage.empty:
                    peak_month = monthly_usage.loc[monthly_usage['total_usage'].idxmax()]
                else:
                    peak_month = {}

                yearly_totals = app_data.groupby('year')['usage'].sum()

                # Seasonal analysis (ensure season exists)
                if 'season' in app_data.columns:
                    season_usage = app_data.groupby('season')['usage'].mean().round(3).to_dict()
                else:
                    season_usage = {}

                # Weekend vs Weekday usage
                weekend_usage = app_data.groupby('is_weekend')['usage'].mean().round(3).to_dict()

                # Holiday usage
                holiday_usage = app_data.groupby('is_holiday')['usage'].mean().round(3).to_dict()

                # Top appliances during specific conditions (e.g., by season)
                if 'season' in app_data.columns:
                    top_appliances_by_season = (
                        app_data.groupby(['season', 'appliance'])['usage']
                        .sum()
                        .sort_values(ascending=False)
                        .groupby(level=0)
                        .head(3)
                        .reset_index()
                        .groupby('season')
                        .apply(lambda x: x[['appliance', 'usage']].to_dict('records'))
                        .to_dict()
                    )
                else:
                    top_appliances_by_season = {}

                # Outliers analysis
                outlier_count = app_data['outlier'].sum()
                outlier_percentage = (outlier_count / len(app_data)) * 100 if len(app_data) > 0 else 0

                # Weather analysis
                weather_summary = {
                    "Average Temperature (°C)": app_data["temp"].mean().round(2) if "temp" in app_data else "N/A",
                    "Average Wind Speed (m/s)": app_data["wspd"].mean().round(2) if "wspd" in app_data else "N/A",
                    "Average Humidity (%)": app_data["rhum"].mean().round(2) if "rhum" in app_data else "N/A",
                    "Average Pressure (hPa)": app_data["pres"].mean().round(2) if "pres" in app_data else "N/A"
                }

                # Add insights to context
                context.append(f"""
                City: {city}, Appliance: {appliance}

                Monthly Usage:
                - {monthly_usage.to_dict('records')}

                Peak Month:
                - {peak_month.get('month_name', 'N/A')} {peak_month.get('year', 'N/A')}: {peak_month.get('total_usage', 'N/A')} kWh

                Yearly Totals:
                - {yearly_totals.to_dict()}

                Seasonal Usage Averages:
                - {season_usage}

                Top Appliances by Season:
                - {top_appliances_by_season}

                Weekend vs Weekday Usage:
                - {weekend_usage}

                Holiday Usage Patterns:
                - {holiday_usage}

                Outlier Information:
                - Total Outliers: {outlier_count}
                - Outlier Percentage: {outlier_percentage:.2f}%

                Weather Summary:
                - {weather_summary}
                """)

        return "\n".join(context)

    def initialize_qa_chain(self):
        """Initialize the question-answering chain using AI tools."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(self.generate_context(self.df))
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        vector_store = FAISS.from_texts(texts, embeddings)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
        )

    def get_analysis(self, question, filtered_df, appliances, cities, date_range):
        """Answer user queries by analyzing the context and filtered data."""
        try:
            # Ensure context is generated for the filtered data
            context = self.generate_context(filtered_df)

            # Explicitly include city details in the query
            city_details = ", ".join(cities) if cities else "all cities"
            query = f"""
            Analyze energy usage for {', '.join(appliances)} in {city_details} 
            from {date_range[0]} to {date_range[1]}.

            Note: The term 'appliance' used in this analysis may refer to either an actual appliance (e.g., refrigerator, heater) or a specific room (e.g., living room, kitchen). Please account for both interpretations in the analysis.
            
            Relevant data and patterns:
            {context}

            User question: {question}

            Please provide a concise summary of the findings, and where possible, organize the results in a table format. Highlight key numbers, trends, and insights in a structured manner for easy understanding.
            """
            
            # Initialize QA chain
            qa_chain = self.initialize_qa_chain()
            response = qa_chain.run(query)
            
            return response
        except Exception as e:
            raise Exception(f"Analysis generation failed: {str(e)}")

class ComsumptionAnalyzer:
    def __init__(self, model_path, past_data):
        self.weather_data = None 
        self.model_path = model_path    
        self.past_data = past_data 

    def return_prediction(self, name):
        model = joblib.load(f'{self.model_path}/{name}/best_model_{name}.joblib')
        return model.predict(self.predict_data)

    def fetch_weather_forecast(self, city, date_str):
        date = datetime.strptime(date_str, '%Y-%m-%d')

        # Extract year and month
        year = date.year
        month = date.month

        # Determine the number of days in the given month
        num_days = calendar.monthrange(year, month)[1]
        month_map = {1: '01', 2: '02', 3: '03', 4: '04', 5: '05', 6: '06', 
                            7: '07', 8: '08', 9: '09', 10: '10', 11: '11', 12: '12'}

        start_date = f"{year}-{month_map[month]}-01"
        end_date = pd.to_datetime(start_date) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        end_date = end_date.strftime('%Y-%m-%d')

        # Assume city variable is defined somewhere in your code
        if city == 'Austin':
            longitude = -97.7431
            latitude = 30.2672
        else:
            longitude = -74.0060
            latitude = 40.7128

        city_name = 'austin' if city == 'Austin' else 'ny'
        print(start_date)
        # Retrieve weather data for the entire month
        hourly = HourlyHistorical()
        options = HistoricalOptions(latitude, longitude, start_date=start_date, end_date=end_date)
        mgr = OpenMeteo(options, hourly.all())
        hourly_data = mgr.get_pandas()

        # Use the manager to get all pertinent data for the month
        hourly_data = mgr.get_pandas()
        # Calculate the daily mean data
        hourly_data.index = pd.to_datetime(hourly_data.index)  # Ensure the index is a DateTimeIndex if not already

        # Group by the date component of the index to calculate daily means
        mean_data = hourly_data.groupby(hourly_data.index.date).mean().reset_index()
        mean_data['city'] = city_name
        mean_data.rename(columns={'index': 'date'}, inplace=True)

        monthly_weather_data = mean_data[['date', 'temperature_2m', 'relativehumidity_2m', 'windspeed_10m', 'surface_pressure', 'city']]
        monthly_weather_data.columns = ['date', 'temp', 'rhum', 'wspd', 'pres', 'city']
        self.weather_data = monthly_weather_data
        # self.weather_data = pd.read_csv('monthly_weather_data.csv')

    def fetch_historical_temperature_averages(self, city, selected_date):
        # Extract the month from the selected_date
        selected_month = pd.to_datetime(selected_date).month
        
        # Define coordinates based on the city
        if city == 'Austin':
            longitude = -97.7431
            latitude = 30.2672
        else:
            longitude = -74.0060
            latitude = 40.7128
        
        # Get the previous three months using a rolling window
        if selected_month in [1, 2, 3]:
            months = [(m - 3) % 12 for m in range(selected_month, selected_month + 3)]
            months = [(12 if m == 0 else m) for m in months] # Adjust for any months identified as 0
        else:
            months = [selected_month - i for i in range(1, 4)]

        # Map numeric months to string representations
        month_map = {1: '01', 2: '02', 3: '03', 4: '04', 5: '05', 6: '06', 
                    7: '07', 8: '08', 9: '09', 10: '10', 11: '11', 12: '12'}
        string_months = [month_map[m] for m in months]

        # Define the years of interest
        years = [2018, 2019]
        
        # Prepare data storage
        
        temperature_data = {f'Month-{i}': [] for i in range(3)}
        # Loop through each year and month to fetch and average the temperature data
        for year in years:
            for i, month in enumerate(string_months):
                # Define start and end dates for the month
                start_date = f"{year}-{month}-01"
                end_date = pd.to_datetime(start_date) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                end_date = end_date.strftime('%Y-%m-%d')
                
                # Fetch historical temperature data
                hourly = HourlyHistorical()
                options = HistoricalOptions(latitude, longitude, start_date=start_date, end_date=end_date)
                mgr = OpenMeteo(options, hourly.all())
                hourly_data = mgr.get_pandas()
                
                # Calculate mean temperature for the month and append to corresponding list
                mean_temp = hourly_data['temperature_2m'].mean()
                temperature_data[f'Month-{i}'].append(mean_temp)
        # Calculate averages for each month based on the collected data
        # mean_temperatures = {key: sum(values) / len(values) for key, values in temperature_data.items()}
        
        return temperature_data

    def return_weather(self):
        return self.weather_data

    def get_date_data(self, date):
        us_holidays = holidays.US()
        data = []
        
        datetime = pd.Timestamp(date)
        month = datetime.month
        month_name = datetime.strftime('%B')
        day_name = datetime.day_name()
        hour = datetime.hour

        is_weekend = datetime.weekday() >= 5  # Saturday is 5, Sunday is 6
        if datetime in us_holidays or is_weekend:
            is_holiday = 1
        else:
            is_holiday = 0
        season = ''
        if month in [12, 1, 2]:
            season = 1
        elif month in [3, 4, 5]:
            season = 2
        elif month in [6, 7, 8]:
            season = 3
        elif month in [9, 10, 11]:
            season = 4

        # Append each row as a dictionary
        data.append({
            'month': month_name,
            'day_name': datetime.weekday(),
            'hour': hour,
            'is_holiday': is_holiday,
            'season': season
        })
        df = pd.DataFrame(data)
        self.date_data = df

    def get_predict_dataset(self):
        input_subset = self.date_data[['month', 'day_name', 'hour', 'season', 'is_holiday']]
        weather_subset = self.weather_data[['temp', 'rhum', 'wspd', 'city']]

        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

        cities = ['austin', 'ny']

        combined_df = pd.concat([input_subset, weather_subset], axis=1)
        categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
        category_definitions = {
            'month': months,
            'city': cities,
        }

        # Create a list of categories for the OneHotEncoder
        categories_list = [category_definitions[col] for col in categorical_cols]
        # Create a preprocessor pipeline
        categorical_transformer = OneHotEncoder(
            categories=categories_list,  # Correctly specify categories
            handle_unknown='ignore'
        )
		   
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),   # Scale numerical features
                ('cat', categorical_transformer, categorical_cols)   # One-hot encode categorical features
            ]
        )
        # Apply transformations to the features
        X_preprocessed = preprocessor.fit_transform(combined_df)
        X_preprocessed = np.nan_to_num(X_preprocessed, nan=0.0)
        self.predict_data = X_preprocessed

    def get_predict(self, city, date, appliances, inputted_r2_score = 0.65):
        self.fetch_weather_forecast(city, date)
        self.get_date_data(date)
        self.get_predict_dataset()
        datetime = pd.Timestamp(date).tz_localize(None)
        xaxis = []
        result = []
        for appliance in appliances:
            # Ensure the 'date' column is also without timezone info
            self.past_data['date'] = pd.to_datetime(self.past_data['date']).dt.tz_localize(None)
            
            city_name = 'austin' if city == 'Austin' else 'ny'
            
            # Calculate the whole month's real usage value
            start_of_month = datetime.replace(day=1)
            end_of_month = (start_of_month + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
            real_value = self.past_data[(self.past_data['city'] == city_name) &
                            (self.past_data['appliance'] == appliance) &
                            (self.past_data['date'] >= start_of_month) &
                            (self.past_data['date'] <= end_of_month)]
            predicted_value = self.return_prediction(appliance)
            # Convert the date for grouping by day
            real_value['day'] = real_value['date'].dt.day

            # Create a complete range of days for the month
            complete_days = pd.DataFrame({'day': range(1, end_of_month.day + 1)})
            # Calculate daily mean usage
            daily_mean_usage = real_value.groupby('day')['usage'].mean().reset_index()
            # Merge the complete range of days with the calculated daily means
            print(daily_mean_usage)
            complete_usage = complete_days.merge(daily_mean_usage, on='day', how='left')

            # Fill missing values with 0
            complete_usage['usage'] = complete_usage['usage'].fillna(0)

            # Convert the usage column to a list
            mean_value_for_month = complete_usage['usage'].tolist()

            if appliance == 'grid':
                mean_value_for_month = [value / 10 for value in mean_value_for_month]
                predicted_value = [value / 10 for value in predicted_value]
            print(mean_value_for_month)
            print(predicted_value)
            r2 = r2_score(mean_value_for_month, predicted_value)
            print(appliance)
            print(r2)
            if len(daily_mean_usage) == len(predicted_value) and r2 > -50:
                print(appliance)
                continue
            
            # Previous months' data extraction
            # Previous month
            target_month = start_of_month.month
            one_month_ago = (start_of_month - pd.DateOffset(months=1)).month
            two_months_ago = (start_of_month - pd.DateOffset(months=2)).month
            three_months_ago = (start_of_month - pd.DateOffset(months=3)).month

            data_last_month = self.past_data[(self.past_data['city'] == city_name) &
                                            (self.past_data['appliance'] == appliance) &
                                            (self.past_data['date'].dt.month == one_month_ago)]['monthly_avg'].mean()

            data_two_months_ago = self.past_data[(self.past_data['city'] == city_name) &
                                                (self.past_data['appliance'] == appliance) &
                                                (self.past_data['date'].dt.month == two_months_ago)]['monthly_avg'].mean()

            data_three_months_ago = self.past_data[(self.past_data['city'] == city_name) &
                                                (self.past_data['appliance'] == appliance) &
                                                (self.past_data['date'].dt.month == three_months_ago)]['monthly_avg'].mean()

            print(data_last_month)

            xaxis.append(appliance + ' 3 months ago')
            xaxis.append(appliance + ' 2 months ago')
            xaxis.append(appliance + ' 1 month ago')
            xaxis.append(appliance + ' Prediction')
            result.append(data_three_months_ago)
            result.append(data_two_months_ago)
            result.append(data_last_month)
            result.append(sum(predicted_value) / len(predicted_value))

        data_dict = {
            'date': date,
            'appliance': xaxis,
            'result': result
        }
        predictions_df = pd.DataFrame(data_dict)
        print(predictions_df)
        return predictions_df 
def main():
    """
    Main entry point for the Streamlit dashboard.
    
    This function sets up the layout, handles data loading, applies user-defined filters, 
    generates visualizations, and provides AI-driven insights.
    """
    st.set_page_config(layout="wide")
    st.title("⚡ WattWise: Empower Your Energy Decisions")
    custom_styles = load_custom_css('custom.css')
    st.markdown(f"<style>{custom_styles}</style>", unsafe_allow_html=True)
    # Load and initialize data
    df = DataLoader.load_and_process_data()
    if df is None:
        st.error("Failed to load data. Please check your data file.")
        return
    
    # Initialize components
    insight_gen = InsightGenerator(df)
    
    # Sidebar filters
    with st.sidebar:
        # City filter: Show all cities as default
        cities = st.multiselect(
            "Select Cities", 
            options=sorted(df['city'].unique()),  # Available options
            default=sorted(df['city'].unique())   # Default to all cities
        )

        # Get the dynamic date range based on selected cities
        min_date, max_date = DataFilter.get_city_date_range(df, cities)
        
        # Date range picker
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),  # Default range for selected cities
            key="date_range"
        )
        
        # Appliance filter: Show all appliances as default
        appliances = st.multiselect(
            "Select Appliances",
            options=sorted(df['appliance'].unique()),  # Available options
            default=sorted(df['appliance'].unique())   # Default to all appliances
        )
    
    # Apply filters
    filtered_df = DataFilter.apply_filters(
        df, date_range[0], date_range[1], cities, appliances
    )
    
    # Display visualizations and insights
    if not filtered_df.empty:
        # Tabs for different sections
        tabs = st.tabs([
        "Overview and Key Insights",
        "Energy Usage Trends",
        "Outliers Analysis",
        "WattBot: Chat with an Expert",
		"Comsumption Predict"
        ])
        
        # Overview and Key Insights Section
        with tabs[0]:
            st.header("Overview and Key Insights")

            # KPI Cards (Row 1)
            st.markdown("### **_Key Metrics_**")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                avg_daily_consumption = filtered_df[filtered_df["appliance"] == "grid"].groupby("day")["usage"].mean().mean()
                st.metric("Avg. Daily Consumption (Grid)", f"{avg_daily_consumption:.2f} kWh")
            with kpi_col2:
                num_appliances = filtered_df["appliance"].nunique()
                st.metric("Number of Appliances", num_appliances)
            with kpi_col3:
                avg_outliers = filtered_df["outlier"].mean()
                st.metric("Avg. Outliers Count", f"{avg_outliers:.2f}")
            with kpi_col4:
                total_data_points = len(filtered_df)
                st.metric("Data Points", total_data_points)

            # Second Row Charts
            st.markdown("### **_Monthly and Appliance Insights_**")
            row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])


            # Chart 1: Grid Usage (Last Month vs Previous Month)
            with row2_col1:
                # Determine the unique months in the filtered dataset
                unique_months = filtered_df["date"].dt.to_period("M").unique()
                
                if len(unique_months) >= 2:
                    # Sort months and select the last two
                    sorted_months = sorted(unique_months)
                    last_month = sorted_months[-1]
                    prev_month = sorted_months[-2]
                    
                    # Filter data for the last two months
                    last_month_data = filtered_df[filtered_df["date"].dt.to_period("M") == last_month]
                    prev_month_data = filtered_df[filtered_df["date"].dt.to_period("M") == prev_month]
                    
                    # Extract day of the month for a common X-axis
                    last_month_data["day_of_month"] = last_month_data["date"].dt.day
                    prev_month_data["day_of_month"] = prev_month_data["date"].dt.day

                    # Label the months for the legend
                    last_month_data["label"] = f"{last_month.start_time.strftime('%B %Y')}"
                    prev_month_data["label"] = f"{prev_month.start_time.strftime('%B %Y')}"

                    # Combine data for plotting
                    combined_data = pd.concat([last_month_data, prev_month_data])
                    combined_data = combined_data.groupby(["day_of_month", "label"])["usage"].sum().reset_index()

                    # Create the line chart
                    fig_usage = px.line(
                        combined_data,
                        x="day_of_month",
                        y="usage",
                        color="label",
                        title="Grid Usage: Last Month vs Previous Month",
                        labels={"day_of_month": "Day of Month", "usage": "Total Usage (kWh)", "label": "Month"},
                    )
                    fig_usage.update_layout(
                        xaxis=dict(title="Day of Month"),
                        yaxis=dict(title="Total Usage (kWh)"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_usage, use_container_width=True)
                else:
                    st.warning("Not enough data to compare two months. Please adjust your date range.")
                
            # Chart 2: Top 5 Appliances by Total Monthly Usage
            with row2_col2:
                appliance_usage = filtered_df[filtered_df["appliance"] != "grid"].groupby("appliance")["usage"].sum().reset_index()
                top_appliances = appliance_usage.sort_values(by="usage", ascending=False).head(5)
                fig_appliances = px.bar(
                    top_appliances,
                    x="usage",
                    y="appliance",
                    orientation="h",
                    title="Top 5 Appliances (Monthly Avg Usage)",
                    labels={"appliance": "Appliance", "usage": "Total Usage (kWh)"},
                )
                fig_appliances.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_appliances, use_container_width=True)

            # Chart 3: Weekend Effect on Grid Consumption
            with row2_col3:
                weekend_usage = filtered_df.groupby("is_weekend")["usage"].mean().reset_index()
                fig_weekend = px.bar(
                    weekend_usage,
                    x="is_weekend",
                    y="usage",
                    title="Weekend Effect on Grid Consumption",
                    labels={"is_weekend": "Weekend", "usage": "Daily Avg Usage (kWh)"},
                )
                fig_weekend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_weekend, use_container_width=True)

            # Third and Fourth Row Chart
            st.markdown("### **_Total Grid Consumption by City (Smoothed)_**")
            filtered_df["day"] = filtered_df["date"].dt.date
            daily_city_usage = (
                filtered_df[filtered_df["appliance"] == "grid"]
                .groupby(["day", "city"])["usage"]
                .mean()
                .reset_index()
            )

            fig_city_usage = px.line(
                daily_city_usage,
                x="day",
                y="usage",
                color="city",
                title="Total Grid Consumption by City (Daily Average)",
                labels={"day": "Date", "usage": "Daily Avg Usage (kWh)", "city": "City"},
            )
            fig_city_usage.update_layout(
                xaxis=dict(title="Date"),
                yaxis=dict(title="Daily Avg Usage (kWh)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_city_usage, use_container_width=True)


        # Energy Usage Trends
        with tabs[1]:
            st.header("Energy Usage Trends")

            # Hourly Usage Line Plot (Subplots by Appliance)
            st.markdown("### **_Hourly Usage Trends_**")

            # Filter data for the selected appliances
            hourly_filtered_df = filtered_df[filtered_df["appliance"].isin(appliances)]

            # Create subplots for each appliance
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            # Create a subplot figure with one row per appliance
            fig_hourly = make_subplots(
                rows=len(appliances),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,  # Compact spacing between subplots
                subplot_titles=[f"Hourly Usage for {appliance.capitalize()}" for appliance in appliances]
            )

            for idx, appliance in enumerate(appliances):
                appliance_data = hourly_filtered_df[hourly_filtered_df["appliance"] == appliance]
                avg_hourly_usage = appliance_data.groupby("hour")["usage"].mean().reset_index()

                fig_hourly.add_trace(
                    go.Scatter(
                        x=avg_hourly_usage["hour"],
                        y=avg_hourly_usage["usage"],
                        mode="lines",
                        name=appliance.capitalize()
                    ),
                    row=idx + 1,
                    col=1
                )

            fig_hourly.update_layout(
                height=250 * len(appliances),  # Adjust height dynamically
                title="Hourly Usage Trends by Appliance",
                xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),  # Increment by 1
                yaxis_title="Average Usage (kWh)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False  # Disable legend since titles show appliance names
            )

            st.plotly_chart(fig_hourly, use_container_width=True, key="hourly_chart_subplots")

            # Heatmap: Appliances vs Usage
            st.markdown("### **_Appliances vs Usage Heatmap_**")

            # Adjust X-axis based on the global filtered data
            heatmap_data = filtered_df.groupby(["appliance", "hour"])["usage"].mean().reset_index()

            # Create the heatmap
            fig_heatmap = px.density_heatmap(
                heatmap_data,
                x="hour",
                y="appliance",
                z="usage",
                color_continuous_scale="Viridis",
                title=f"Appliances vs Hour (Usage)",
                labels={"hour": "Hour", "appliance": "Appliance", "usage": "Avg Usage (kWh)"},
            )
            fig_heatmap.update_layout(
                xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),  # Increment by 1
                yaxis=dict(title="Appliance"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=800  # Make the heatmap larger
            )

            # Display the heatmap
            st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_chart") 
        
        
        with tabs[2]:

            st.header("Outliers Analysis")

            # Section 1: Outlier Count by Appliance
            st.markdown("### **_Outlier Count by Appliance_**")

            # Group by appliance to calculate total outliers
            outlier_count_by_appliance = (
                filtered_df.groupby("appliance")["outlier"].sum().reset_index()
            )
            outlier_count_by_appliance.rename(columns={"outlier": "outlier_count"}, inplace=True)

            # Bar chart for outlier count by appliance
            fig_outlier_count = px.bar(
                outlier_count_by_appliance,
                x="appliance",
                y="outlier_count",
                title="Total Outliers by Appliance",
                labels={"appliance": "Appliance", "outlier_count": "Outlier Count"},
                color="outlier_count",
                color_continuous_scale="Viridis"
            )
            fig_outlier_count.update_layout(
                xaxis=dict(title="Appliance"),
                yaxis=dict(title="Outlier Count"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_outlier_count, use_container_width=True, key="outlier_count_chart")

            # Section 2: Heatmap of Outliers by Appliance and Hour
            st.markdown("### **_Outliers Heatmap by Appliance and Hour_**")

            # Group by appliance and hour to calculate total outliers
            outlier_heatmap_data = (
                filtered_df.groupby(["appliance", "hour"])["outlier"]
                .sum()
                .reset_index()
                .rename(columns={"outlier": "outlier_count"})
            )

            # Heatmap for outliers by appliance and hour
            fig_outlier_heatmap = px.density_heatmap(
                outlier_heatmap_data,
                x="hour",
                y="appliance",
                z="outlier_count",
                color_continuous_scale="Viridis",
                title="Heatmap of Outliers by Appliance and Hour",
                labels={"hour": "Hour of Day", "appliance": "Appliance", "outlier_count": "Outlier Count"},
            )
            fig_outlier_heatmap.update_layout(
                xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
                yaxis=dict(title="Appliance"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=600
            )
            st.plotly_chart(fig_outlier_heatmap, use_container_width=True, key="outlier_heatmap_chart")

            # Section 3: Outlier Trends by Day and Month
            st.markdown("### **_Outlier Trends by Day and Month_**")

            # Outliers by day of the week
            outliers_by_day = (
                filtered_df.groupby(filtered_df["date"].dt.day_name())["outlier"]
                .mean() * 100
            ).reset_index().rename(columns={"date": "day_name", "outlier": "outlier_percentage"})

            # Outliers by month
            outliers_by_month = (
                filtered_df.groupby(filtered_df["date"].dt.month_name())["outlier"]
                .mean() * 100
            ).reset_index().rename(columns={"date": "month", "outlier": "outlier_percentage"})

            # Bar charts for trends
            col1, col2 = st.columns(2)
            with col1:
                fig_day = px.bar(
                    outliers_by_day,
                    x="day_name",
                    y="outlier_percentage",
                    title="Outlier Percentage by Day of the Week",
                    labels={"day_name": "Day of the Week", "outlier_percentage": "Outlier Percentage (%)"}
                )
                fig_day.update_layout(
                    xaxis=dict(title="Day of the Week"),
                    yaxis=dict(title="Outlier Percentage (%)"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_day, use_container_width=True, key="outlier_by_day_chart")

            with col2:
                fig_month = px.bar(
                    outliers_by_month,
                    x="month",
                    y="outlier_percentage",
                    title="Outlier Percentage by Month",
                    labels={"month": "Month", "outlier_percentage": "Outlier Percentage (%)"}
                )
                fig_month.update_layout(
                    xaxis=dict(title="Month"),
                    yaxis=dict(title="Outlier Percentage (%)"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_month, use_container_width=True, key="outlier_by_month_chart")
            
        with tabs[3]:
            st.header("WattBot: Chat with an Expert")
            question = st.text_input(
                "Ask about your energy usage:",
                "What are the main patterns and saving opportunities?"
            )
            if st.button("Generate Response"):
                if question:
                    with st.spinner("Analyzing..."):
                        try:
                            analysis = insight_gen.get_analysis(
                                question,
                                filtered_df,
                                appliances,
                                cities,
                                date_range
                            )
                            st.write(analysis)
                        except Exception as e:
                            st.error(f"Error generating analysis: {str(e)}")
                            st.error(str(e))
                else:
                    st.warning("Please enter a question before generating a response.")
        # Energy Consumption Section
        with tabs[4]:
            st.header("Consumption Predict")
            selected_city = st.selectbox('Select a City:', ['Austin', 'NewYork'])

            selected_appliances = st.multiselect(
                "Select Appliances",
                options=sorted(df['appliance'].unique()),  # Available options
                default=sorted(df['appliance'].unique()),   # Default to all appliances
                key="consumption-appliance"
            )

            default_date = "2018-12-31"

            # Convert to a datetime object
            date_object = datetime.strptime(default_date, "%Y-%m-%d")

            # Get today's date
            today = datetime.today().date()

            # Create a date input with max_value set to today's date
            selected_date = st.date_input(
                'Select a date:',
                date_object.date(),
                max_value=today
            )

            # r2_score = st.slider(
            #     label="Select R2 Score",
            #     min_value=0.1,
            #     max_value=0.99,
            #     value=0.6,  # Default starting value
            #     step=0.05
            # )

            if st.button('Predict'):
                city = selected_city
                comsumption_analyzer = ComsumptionAnalyzer(model_path = model_path, past_data = df)
                # Get predictions
                selected_date_str = selected_date.strftime('%Y-%m-%d')
                predictions_df = comsumption_analyzer.get_predict(city, selected_date_str, selected_appliances, 0.65)
                average_temperature = comsumption_analyzer.fetch_historical_temperature_averages(city, selected_date)
                # Display metrics within a container
                with st.container(key="key-2018-container"):
                    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

                    with kpi_col1:
                        temperature = round(average_temperature['Month-2'][0], 2)
                        st.metric("3 Months Ago Temperature in 2018", f"{temperature} °C")

                    with kpi_col2:
                        temperature = round(average_temperature['Month-1'][0], 2)
                        st.metric("2 Months Ago Temperature in 2018", f"{temperature} °C")

                    with kpi_col3:
                        temperature = round(average_temperature['Month-0'][0], 2)
                        st.metric("1 Months Ago Temperature in 2018", f"{temperature} °C")

                with st.container(key="key-2019-container"):
                    with kpi_col1:
                        temperature = round(average_temperature['Month-2'][1], 2)
                        st.metric("3 Months Ago Temperature in 2019", f"{temperature} °C")

                    with kpi_col2:
                        temperature = round(average_temperature['Month-1'][1], 2)
                        st.metric("2 Months Ago Temperature in 2019", f"{temperature} °C")

                    with kpi_col3:
                        temperature = round(average_temperature['Month-0'][1], 2)
                        st.metric("1 Months Ago Temperature in 2019", f"{temperature} °C")

                # Display the daily average graph within a container
                with st.container(key="key-daily_average-graph"):
                    st.markdown("### **_Weather Forecast_**")
                    
                    # Display prediction header
                    st.subheader(f"Prediction for {selected_appliances} in {city} on {selected_date}:")

                    # Prepare data for visualization
                    comparison_df = pd.DataFrame({
                        "Category": predictions_df['appliance'],
                        "Usage (kWh)": predictions_df['result']
                    })

                    # Generate color list based on index
                    colors = ['yellow' if ((i + 1) % 4) == 0 else 'blue' for i in range(len(comparison_df))]
                    
                    # Create and customize the bar chart
                    fig = px.bar(
                        comparison_df,
                        x="Category",
                        y="Usage (kWh)",
                        title="Comparison of Predicted vs Historical Usage",
                        text="Usage (kWh)"
                    )

                    # Update bar colors
                    fig.update_traces(marker=dict(color=colors))

                    # Center the title
                    fig.update_layout(title={'x': 0.5})

                    # Plot the chart
                    st.plotly_chart(fig, use_container_width=True)

                    
if __name__ == "__main__":
    main()