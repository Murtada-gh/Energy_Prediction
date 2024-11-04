# Energy_Prediction
This repository contains a machine learning project that leverages PecanStreet energy data to develop two key solutions: (1) an outlier detection system for identifying anomalous energy consumption, and (2) a predictive model to forecast energy usage for specified periods.

# PecanStreet Energy Data Analysis and Machine Learning Solutions

This project leverages PecanStreet energy data from two selected cities to explore and model energy consumption patterns. The project has two main objectives:
1. **Outlier Detection**: Identify unusual patterns or anomalies in energy consumption data.
2. **Consumption Prediction**: Forecast energy usage for specified periods based on historical data and weather conditions.

By combining energy consumption data with weather data, this project aims to better understand the factors influencing residential energy usage.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This repository contains all necessary files, data, and scripts to train machine learning models for energy data analysis, including:
- **Outlier Detection**: Detecting anomalies in energy consumption to understand unusual patterns or potential data errors.
- **Consumption Prediction**: Building models to forecast future energy consumption, informed by past trends and weather conditions.

This project was created as a foundation for research and development on energy consumption trends, with potential applications for energy efficiency analysis, demand response programs, and smart grid optimization.

## Data

This project uses PecanStreet's anonymized residential energy data, specifically for two selected cities. For each city, the data is divided into:
- **Energy Consumption Data**: Contains the raw energy consumption measurements for residential properties.
- **Weather Data**: Includes historical weather conditions corresponding to the time periods of the energy data, helping establish any relationships between weather and energy usage.

### Data Files

Each data file is provided in CSV format:
- **Raw Data**: Located in `data/raw/` and includes the original files downloaded from PecanStreet.
  - `data/raw/city1_energy.csv` – Raw energy consumption data for City 1
  - `data/raw/city2_energy.csv` – Raw energy consumption data for City 2
  - `data/raw/city1_weather.csv` – Weather data for City 1
  - `data/raw/city2_weather.csv` – Weather data for City 2
- **Processed Data**: Located in `data/processed/` and includes the cleaned and merged data, prepared for modeling.
  - `data/processed/city1_data.csv` – Cleaned and merged data for City 1 (energy + weather)
  - `data/processed/city2_data.csv` – Cleaned and merged data for City 2 (energy + weather)

### Data Source

The energy data is sourced from [PecanStreet](https://www.pecanstreet.org/), a provider of anonymized residential energy data for research purposes.

