# demand_forecasting

This project was developed based on the [Store Sales - Time Series Forecasting Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/leaderboard)
, which focuses on predicting daily sales for thousands of product families across multiple grocery stores from Corporación Favorita, one of the largest retail chains in Ecuador. The challenge involves building forecasting models capable of handling complex retail demand patterns, seasonality, promotions, holidays, and external economic factors.

The objective of this project was to design an end-to-end time series forecasting pipeline combining exploratory data analysis, feature engineering, model experimentation, validation strategies, and forecasting evaluation. The dataset includes transactional sales data, promotional campaigns, store metadata, oil prices, and holiday events, enabling the development of robust machine learning solutions for real-world retail forecasting scenarios.

Throughout the project, several forecasting approaches were explored, including statistical models and machine learning techniques such as gradient boosting and recursive forecasting strategies. Special attention was given to:

- Time series preprocessing and missing date handling
- Lag and rolling-window feature engineering
- Calendar and holiday feature extraction
- Cross-validation for time series
- Model evaluation using forecasting metrics
- Multi-step forecasting pipelines

This repository demonstrates practical applications of demand forecasting and MLOps-oriented experimentation workflows using Python, pandas, PySpark, scikit-learn, and modern forecasting libraries. The project also emphasizes reproducibility, modular code organization, and scalable experimentation practices commonly used in production-grade forecasting systems.

The competition and dataset provide an excellent benchmark for learning and applying forecasting techniques in large-scale retail environments, where accurate demand prediction directly impacts inventory optimization, logistics, and business decision-making.

This project uses [RMSLE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_log_error.html) as the main evaluation metric.
