# Airport Passenger Traffic Prediction in the Dominican Republic

This project uses machine learning to predict passenger traffic at airports in the Dominican Republic, with a focus on understanding seasonal patterns and managing high-traffic periods. The `RandomForestRegressor` model was selected and optimized to accurately forecast traffic across different airports and timeframes, considering seasonal and airport-specific trends.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Analysis and Results](#analysis-and-results)
- [Conclusions](#conclusions)
- [Next Steps](#next-steps)

## Project Description
This project analyzes data on passenger entries and exits at various Dominican airports, aiming to forecast passenger traffic for operational and resource management purposes. The dataset contains information on airport locations, monthly passenger counts, and seasonal traffic variations.

## Installation
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/your-username/project-name.git
cd project-name
pip install -r requirements.txt
```
## Usage
1. Run the script prediccion_trafico_aeropuertos.py to train the model and review the prediction results.
2. For deeper exploration, you can modify the script or open the Jupyter Notebook in the notebooks folder to visualize intermediate results and plots.

# Analysis and Results
## Outlier Detection and Seasonal Patterns
We performed an outlier analysis to identify months with exceptionally high or low passenger traffic. Outliers were often aligned with specific seasons, confirming the need for seasonally-aware predictions. Here are some key findings:

- **High-Traffic Periods:** Outliers observed during peak seasons suggest traffic spikes that require additional resource allocation.
- **Event-Based Patterns:** Significant traffic reductions in certain years could be linked to major events such as the 2020 pandemic.

## Model Performance
The model was evaluated with and without log transformation to manage the high variability in traffic data:

- **Mean Absolute Error (MAE) without log transformation:** Value
- **Mean Squared Error (MSE) without log transformation:** Value
- **MAE with log transformation:** Value
- **MSE with log transformation:** Value

The log transformation improved the model’s handling of large values, providing more accurate predictions during high-traffic periods.

# Conclusions

The seasonal patterns observed in outliers suggest a consistent increase in traffic during certain months, supporting the use of seasonality as a predictive feature. The model, optimized with hyperparameter tuning and log transformation, demonstrates improved accuracy, particularly for high-traffic periods.

