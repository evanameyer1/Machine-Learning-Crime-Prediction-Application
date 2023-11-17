# ML Crime Prediction
---
### Introduction
Project Overview: The Machine Learning Crime Prediction Application is a data-driven initiative aimed at exploring machine learning models and frameworks by predicting crime in Chicago.

Forecasting both violent and non-violent crimes across 77 neighborhoods for a future one-year period posed a significant challenge due to the scale and complexity of the dataset. We worked with 750 million rows of Chicago Crime and Infrastructure data, necessitating thorough cleaning and preprocessing.

Our Solution Approach:

Developed multivariate time series adaptations of models such as Linear Regression, LSTM, Auto Regression, Moving Average, and their combinations to enhance performance and robustness.
Constructed a multivariate multi-step time series adaptation of a Random Forest model trained on three years of data to predict future crime occurrences with high accuracy.

#### Tech Stack:
- Python
- Pandas
- PyTorch
- scikit-learn (SKlearn)
- ARiMA
- Matplotlib
- Shapely
---
### Description of Data

#### Crime Data:
- Includes crime type, location details, arrest status, and time of occurrence.
- Encompasses attributes such as primary crime type, location descriptions, district details, and coordinates (latitude and longitude).
- Offers insights into crime patterns and trends in Chicago.

#### Vacant and Abandoned Buildings:
- Focuses on abandoned buildings in Chicago.
- Provides details on building location, occupancy status, potential hazards, and observations related to property use.

#### Divvy Bike Trips in Chicago:
- Contains trip IDs, start/stop times, station IDs, and station names.
- Aids in understanding Divvy bike usage patterns and movement across the city.

#### Bike Stations in Chicago:
- Presents information about bike station locations.
- Provides latitude and longitude coordinates for each station, aiding in analyzing station distribution.

#### Bus Stops in Chicago:
- Details bus stop IDs, stop names, and geographical coordinates.
- Offers insights into the city's public transportation infrastructure.

#### Train Stations in Chicago:
- Includes station numbers, names, and geographical coordinates.
- Provides information about the city's railway network and transportation hubs.

#### Alleylights Out in Chicago:
- Records outage start/end dates, duration, and geographical coordinates.
- Focuses on lighting disruptions affecting Chicago's alleys.

#### Streetlights Out in Chicago:
- Records creation/completion dates, service request types, addresses, and geographical coordinates.
- Offers insights into areas experiencing street lighting issues in the city.
---
### Data Cleaning
When approaching this project, we had anticipated data cleaning to be the most labor intensive aspect. Despite the fact that prior to approaching machine learning, we only had experience with similar data cleaning and utilization projects, the complexity and size of the dataset provided us with many challenges in terms of problems and algorithm runtime.

Here are the major steps that we took during cleaning:
---
### Research and Preparation
This was by far the most time-intensive aspect of the project, as neither of us had done anything in machine learning before, let alone anything on this scale. Despite this, we dove headfirst into the high level applications of machine learning for time-series analysis. Unsurprisingly, we ran into problem after problem, and after realizing that most of our problems were due to a lack of understanding, we started from the basics.

#### Time Series Machine Learning Models
Of the machine learning models that we researched, the ones that seemed most applicable and attractive for our use case were the ones in the ARIMA suite. However, in application, they still showed some drawbacks in terms of accuracy, and weren't able to model the true correlation between the different aspects of our final dataset.

Aside from ARIMA, we explored LSTM, Random Forest, Linear Regression, and NN's for time-series application, all of which had similar issues ARIMA.

At this point, our interest in machine learning had grown to the point that we wanted to actually write many of these models from scratch, so we started from the basics and began working up.

We decided to do a ground-up implementation of each of these models but with the context of our use case in mind the entire time, hopefully achieving a strong representation of the correlation between Chicago crime and infrastructure.

---
