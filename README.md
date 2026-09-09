# electrical-energy
# Household Electric Power Consumption Forecasting

## Overview

This project focuses on forecasting household electric power consumption using historical time series data. The goal is to build and evaluate predictive models that can accurately estimate future energy usage based on past observations. The work includes data preprocessing, exploratory analysis, feature engineering, model training, and performance evaluation.

## Dataset

The dataset used is the [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from the UCI Machine Learning Repository.

- **Source:** UCI Machine Learning Repository
- **Description:** Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.
- **Attributes:**
  - `Date` and `Time`
  - `Global_active_power`
  - `Global_reactive_power`
  - `Voltage`
  - `Global_intensity`
  - `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`
- **Missing values:** Represented as `?`, require handling during preprocessing.

