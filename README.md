# ML Crime Prediction
---
## Introduction
Project Overview: The Machine Learning Crime Prediction Application is a data-driven initiative aimed at exploring machine learning models and frameworks by predicting crime in Chicago.

Forecasting both violent and non-violent crimes across 77 neighborhoods for a future one-year period posed a significant challenge due to the scale and complexity of the dataset. We worked with 750 million rows of Chicago Crime and Infrastructure data, necessitating thorough cleaning and preprocessing.

### Our Solution Approach:

Our goal was to develop a robust multivariate multi-step time-series model for our dataset. Despite an extensive search for a publicly available, rigorously validated, and well-documented model tailored to our specific use case, we found none. Consequently, we decided to adapt prominent models, listed below, to handle and forecast multivariate multi-step time-series data. We then conducted a comparative analysis of the resulting accuracies for each model.

During the adaptation process, we also conceived a novel model approach, detailed at the bottom of the README, which we anticipate will exhibit superior predictive capabilities for our dataset. Currently, we are in the midst of constructing, testing, and validating this model. Stay tuned for further updates!

### Tech Stack:
- Python
- Pandas
- Numpy
- PyTorch
- Lightning
- scikit-learn (SKlearn)
- Statsmodels
- ARiMA
- Matplotlib
- Shapely

### Models:
- Linear Regression
- LSTM
- Auto Regression
- Moving Average
- Random Forest
- Custom LSTM to NN model
- Additional combinations of ARIMA, SARIMA, VAR, and AR
---
## Description of Data

### Crime Data:
- Includes crime type, location details, arrest status, and time of occurrence.
- Encompasses attributes such as primary crime type, location descriptions, district details, and coordinates (latitude and longitude).
- Offers insights into crime patterns and trends in Chicago.

### Vacant and Abandoned Buildings:
- Focuses on abandoned buildings in Chicago.
- Provides details on building location, occupancy status, potential hazards, and observations related to property use.

### Divvy Bike Trips in Chicago:
- Contains trip IDs, start/stop times, station IDs, and station names.
- Aids in understanding Divvy bike usage patterns and movement across the city.

### Bike Stations in Chicago:
- Presents information about bike station locations.
- Provides latitude and longitude coordinates for each station, aiding in analyzing station distribution.

### Bus Stops in Chicago:
- Details bus stop IDs, stop names, and geographical coordinates.
- Offers insights into the city's public transportation infrastructure.

### Train Stations in Chicago:
- Includes station numbers, names, and geographical coordinates.
- Provides information about the city's railway network and transportation hubs.

### Alleylights Out in Chicago:
- Records outage start/end dates, duration, and geographical coordinates.
- Focuses on lighting disruptions affecting Chicago's alleys.

### Streetlights Out in Chicago:
- Records creation/completion dates, service request types, addresses, and geographical coordinates.
- Offers insights into areas experiencing street lighting issues in the city.
---
## Data Cleaning and Wrangling

In this section, we detail the steps taken to clean and wrangle the dataset for the project. Given the complexity and size of the dataset, we encountered challenges that required meticulous handling of missing values, standardizing column names, and converting locational data formats.

### 1. Standard Cleaning Process

We implemented a standardized cleaning process to ensure consistency in the dataset. This involved creating a `clean_data` function that accepts a DataFrame and a type parameter, performing the following steps:

```python
def clean_data(df, type):
    # Select relevant columns and rename them for consistency
    df = df[['Creation Date', 'Service Request Number', 'Type of Service Request', 'Latitude', 'Longitude']]
    df.rename(columns={'Creation Date': 'date', 'Service Request Number': 'id', 'Type of Service Request': 'type', 'Latitude': 'lat', 'Longitude': 'long'}, inplace=True)
    
    # Calculate the percentage of null values in the 'lat' column
    perc_null = sum(df.lat.isnull()) / len(df)
    
    # Check if the percentage of null values is too large to clean
    if perc_null > 0.005:
        print(f'perc_null of {perc_null} is too large to clean')
    else:
        # If acceptable, drop rows with null values in 'lat' and 'long'
        df.dropna(subset=['lat', 'long'], inplace=True)
        print(f'successfully removed {sum(df.lat.isnull())} nulls or {perc_null}%')
    
    # Sort the DataFrame by the 'date' column and reset the index
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Set the 'type' column to the specified type parameter
    df['type'] = type
    
    # Return the cleaned DataFrame
    return df
```

### 2. Clean Data Method (Linear Regression Class)

In the Linear Regression class, we applied the clean_data method to both training and testing datasets, converting the 'date' column to datetime format and extracting additional temporal features:

```python
def clean_data(self):
    # Loop through both train and test datasets
    for df in [self.train_dataset, self.test_dataset]:
        # Convert the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract features like hour, day of the week, day of the year, and month from the 'date' column
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        
        # Drop the original 'date' column
        df.drop('date', axis=1, inplace=True)

    # Set the 'length' attribute to the length of the train dataset
    self.length = len(self.train_dataset)

    # Return the cleaned train and test datasets
    return self.train_dataset, self.test_dataset
```

### 3. Convert to Polygon Function

We introduced a convert_to_polygon function that takes a DataFrame with locational data in the form of MULTIPOLYGON and converts it into a list of Polygon objects:

```python
def convert_to_polygon(df):
    # Initialize an empty list to store updated polygons
    updated_polygons = []
    
    # Iterate through each row in the 'the_geom' column
    for row in df['the_geom']:
        # Extract and clean the target string representing the polygon
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        
        # Initialize an empty list to store the final set of coordinates
        final = []
        
        # Iterate through each point in the polygon
        for point in points:
            # Split the point into latitude and longitude
            temp = point.split(' ')
            tup = float(temp[1].replace(')', '').replace('(', '')), float(temp[0].replace(')', '').replace('(', '')) 
            final.append(tup)
        
        # Create a Polygon object from the coordinates and append it to the list
        polygons = Polygon(final)
        updated_polygons.append(polygons)
    
    # Return the list of updated Polygon objects
    return updated_polygons
```

### 4. Determine Within Function

The determine_within function now takes latitude and longitude columns in a DataFrame and determines whether each location is within specified polygons, assigning a status and area accordingly:

```python
def determine_within(df):
    # Initialize lists to store status and areas
    statuses = []
    areas = []
    
    # Iterate through each row in the DataFrame
    for i in range(len(df)):
        # Create a Point object from latitude and longitude
        temp = geometry.Point(df.lat.loc[i], df.long.loc[i])
        
        # Initialize status and iterate through clean reference polygons
        status = 0
        for polygon in clean_reference:
            if polygon.contains(temp):
                status = 1
        statuses.append(status)
        
        # Initialize variables for inside check and iterate through clean areas
        index = 0
        inside = False
        for polygon in clean_areas:
            if polygon.contains(temp):
                areas.append(unclean_areas['AREA_NUMBE'].iloc[index])
                inside = True
            index += 1
        
        # If not inside any clean area, append 0 to the areas list
        if not inside:
            areas.append(0)
    
    # Add 'status' and 'area' columns to the DataFrame
    df['status'] = statuses
    df['area'] = areas
    
    # Return the updated DataFrame
    return df
```

### 5. Prepare Data Method (Random Forest Class)

The prepare_data method in the Random Forest class prepares time-series datasets for modeling by creating various lag, rolling mean, rolling min, rolling max, and rolling standard deviation features for each target variable:

```python
def prepare_data(self, exempt=[]):
    """
    Prepare the datasets for modeling.

    Parameters:
    - exempt: Features to be exempted from preparation.

    Returns:
    - train_df, test_df, features: The prepared training and testing datasets and the list of features.
    """
    # Iterate through each target variable
    for target in self.targets:
        # Iterate through both training and testing datasets
        for df in [self.train_df, self.test_df]:
            # Shift by Date Cycles
            df[f'{target[:2]}_next_hour'] = df[target].shift(-1)
            df[f'{target[:2]}_next_day'] = df[target].rolling(window=24).sum()
            df[f'{target[:2]}_next_weekday'] = df[target].rolling(window=7 * 24).sum()
            df[f'{target[:2]}_next_month'] = df[target].rolling(window=30 * 24).sum()

            # Lag by Date Cycles
            df[f'{target[:2]}_inverse_hour'] = df[target].shift(1)
            df[f'{target[:2]}_inverse_next_day'] = df[target].diff(24)
            df[f'{target[:2]}_inverse_next_weekday'] = df[target].diff(7 * 24)
            df[f'{target[:2]}_inverse_next_month'] = df[target].diff(30 * 24)

            df.dropna(inplace=True)

            # Rolling Mean by Date Cycles
            df[f"{target[:2]}_6hour_mean"] = df[target].rolling(6).mean()
            df[f"{target[:2]}_12hour_mean"] = df[target].rolling(12).mean()
            df[f"{target[:2]}_24hour_mean"] = df[target].rolling(24).mean()
            df[f"{target[:2]}_week_mean"] = df[target].rolling(24*7).mean()
            df[f"{target[:2]}_30day_mean"] = df[target].rolling(24*30).mean()

            # Rolling Min by Date Cycles
            df[f"{target[:2]}_6hour_min"] = df[target].rolling(6).min()
            df[f"{target[:2]}_12hour_min"] = df[target].rolling(12).min()
            df[f"{target[:2]}_24hour_min"] = df[target].rolling(24).min()
            df[f"{target[:2]}_week_min"] = df[target].rolling(24*7).min()
            df[f"{target[:2]}_30day_min"] = df[target].rolling(24*30).min()

            # Rolling Max by Date Cycles
            df[f"{target[:2]}_6hour_max"] = df[target].rolling(6).max()
            df[f"{target[:2]}_12hour_max"] = df[target].rolling(12).max()
            df[f"{target[:2]}_24hour_max"] = df[target].rolling(24).max()
            df[f"{target[:2]}_week_max"] = df[target].rolling(24*7).max()
            df[f"{target[:2]}_30day_max"] = df[target].rolling(24*30).max()

            # Rolling Standard Deviation by Date Cycles
            df[f"{target[:2]}_6hour_std"] = df[target].rolling(6).std()
            df[f"{target[:2]}_12hour_std"] = df[target].rolling(12).std()
            df[f"{target[:2]}_24hour_std"] = df[target].rolling(24).std()
            df[f"{target[:2]}_week_std"] = df[target].rolling(24*7).std()
            df[f"{target[:2]}_30day_std"] = df[target].rolling(24*30).std()

    # Create a list of features, exempting specified features
    self.features = [feature for feature in self.train_df.columns if feature not in self.targets or feature in exempt]

    # Return the prepared training and testing datasets and the list of features
    return self.train_df, self.test_df, self.features
```

### 6. Normalize DataFrame Method (Linear Regression Class)

The normalize_dataframe method normalizes all columns in a pandas DataFrame using the mean and standard deviation calculated from the training dataset:

```python
def normalize_dataframe(self):
    """
    Normalize all columns in a pandas DataFrame.
    
    Returns:
    - A new DataFrame with normalized values.
    """
    # Calculate the mean and standard deviation for each column using the training dataset
    means = self.train_dataset.mean()
    stds = self.train_dataset.std()

    # Normalize both training and test datasets
    for df in [self.train_dataset, self.test_dataset]:
        # Select non-date columns for normalization
        non_date_columns = [col for col in df.columns if col != 'date']  # Adjust the condition based on your date columns

        # Normalize non-date columns
        df[non_date_columns] = (df[non_date_columns] - means[non_date_columns]) / stds[non_date_columns]

    # Return the normalized training and test datasets
    return self.train_dataset, self.test_dataset
```

### 7. Aggregation and Pivoting Examples

#### Fill Gaps Function

This function fills in missing combinations of date, hour, and area with zero values in a DataFrame:

```python
def fill_gaps(df):
    # Create a date range, hour range, and area range
    all_dates = pd.date_range(start='2001-01-01', end='12/31/2022', freq='D')
    all_hours = range(1, 24)
    all_areas = range(1, 78)
    
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Create all possible combinations of date, hour, and area
    date_hour_combinations = pd.DataFrame(([datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d"), hour, area] for date in all_dates for hour in all_hours for area in all_areas), columns=['date', 'hour', 'area'])
    
    # Merge the original DataFrame with the combinations, filling missing values with zero
    merged_df = date_hour_combinations.merge(df, on=['date', 'hour', 'area'], how='outer').fillna(0)

    # Return the merged DataFrame with filled gaps
    return merged_df
```
    
#### Aggregate Data Function

This function aggregates and merges various datasets related to crime, ridership, divvy trips, lighting, and vacant buildings:

```python
def aggregate_data():

    # Clean the crime data for smoother aggregation
    clean_crime['date'] = [datetime.datetime.strptime(clean_crime.date.iloc[i], '%m/%d/%Y %I:%M:%S %p') for i in range(len(clean_crime))]
    clean_crime['hour'] = clean_crime.date.apply(lambda x: (x.hour + 1))
    clean_crime['date'] = clean_crime.date.apply(lambda x: x.date())

    # Define violent crimes and categorize crime types
    violent_crimes = ['BATTERY', 'ASSAULT', 'CRIM SEXUAL ASSAULT', 'SEX OFFENSE', 'WEAPONS VIOLATION',
                      'HOMICIDE', 'KIDNAPPING', 'ROBBERY', 'INTIMIDATION', 'ARSON', 'CRIMINAL SEXUAL ASSAULT', 'HUMAN TRAFFICKING']
    clean_crime['type'] = clean_crime['type'].apply(lambda x: 2 if x in violent_crimes else 1)

    print('clean_crime successfully read in')

    # Group the crime dataset utilizing a count aggregation
    grouped_crime = clean_crime.groupby(['date', 'hour', 'type', 'area']).size().reset_index(name='count')

    # Pivot the violent vs non-violent counts to display across our grouped columns
    grouped_crime = grouped_crime.pivot_table(index=['date', 'hour', 'area'], columns='type', values='count', fill_value=0).reset_index()
    grouped_crime.rename(columns={1: "non-violent", 2: 'violent'}, inplace=True)

    print('crime dataset successfully grouped')

    # Call the fill_gaps function
    grouped_crime = fill_gaps(grouped_crime)

    print('fill gaps function successfully ran in')

    # Merge the area_reference dataset on top of the grouped_crime data
    grouped_crime = grouped_crime.merge(area_reference, left_on='area', right_on='id', how='left')
    grouped_crime.drop(columns=['id'], inplace=True)

    print('area reference data successfully merged')

    # Match datatypes of columns and then merge the two datasets together
    grouped_crime['date'] = pd.to_datetime(grouped_crime['date'])
    clean_ridership['date'] = pd.to_datetime(clean_ridership['date'])
    grouped_crime = grouped_crime.merge(clean_ridership, on=['date', 'area'], how='left')

    print('clean ridership data successfully merged')

    # Match datatypes of columns and then group the dataset to make merging together easier
    clean_divvy_trips['hour'] = pd.to_datetime(clean_divvy_trips['date']).dt.hour
    clean_divvy_trips['date'] = pd.to_datetime(pd.to_datetime(clean_divvy_trips['date']).dt.date)
    grouped_divvy = clean_divvy_trips.groupby(['date', 'hour', 'area'])['station_id'].agg('count').reset_index()

    # Merge the grouped divvy data onto the base crime data
    grouped_crime = grouped_crime.merge(grouped_divvy, on=['date', 'hour', 'area'], how='left')
    grouped_crime.rename(columns={'station_id': 'bike_rides', 'rides': 'train_rides'}, inplace=True)

    print('clean divvy trips data successfully merged')

    # Group lighting data and merge onto the base crime data
    grouped_lighting = clean_lighting.groupby(['date', 'area'])['lat'].agg('count').reset_index()
    grouped_lighting.date = pd.to_datetime(grouped_lighting.date)
    grouped_crime = grouped_crime.merge(grouped_lighting, on=['date', 'area'], how='left')
    grouped_crime.rename(columns={'lat': 'lighting'}, inplace=True)

    print('clean lighting data successfully merged')

    # Group vacancies data and merge onto the base crime data
    clean_vacant_buildings.date = pd.to_datetime(clean_vacant_buildings.date)
    grouped_vacancies = clean_vacant_buildings[clean_vacant_buildings.date <= pd.to_datetime('2022-12-31')].groupby(['date', 'area'])['long'].agg('count').reset_index()
    grouped_crime = grouped_crime.merge(grouped_vacancies, on=['date', 'area'], how='left')
    grouped_crime.rename(columns={'long': 'vacant_buildings'}, inplace=True)

    print('clean vacancies data successfully merged')

    return grouped_crime
```

---
## Research and Preparation
This was by far the most time-intensive aspect of the project, as neither of us had done anything in machine learning before, let alone anything on this scale. Despite this, we dove headfirst into the high level applications of machine learning for time-series analysis. Unsurprisingly, we ran into problem after problem, and after realizing that most of our problems were due to a lack of understanding, we started from the basics.

### Time Series Machine Learning Models
Of the machine learning models that we researched, the ones that seemed most applicable and attractive for our use case were the ones in the ARIMA suite. However, in application, they still showed some drawbacks in terms of accuracy, and weren't able to model the true correlation between the different aspects of our final dataset.

Aside from ARIMA, we explored LSTM, Random Forest, Linear Regression, and NN's for time-series application, all of which had similar issues ARIMA.

At this point, our interest in machine learning had grown to the point that we wanted to actually write many of these models from scratch, so we started from the basics and began working up.

We decided to do a ground-up implementation of each of these models but with the context of our use case in mind the entire time, hopefully achieving a strong representation of the correlation between Chicago crime and infrastructure.

---
## Defining Testing and Training Datasets

### Date Ranges of each Dataset

**1. Crime Dataset**
- *Min:* 01/01/2001 01:00:00 AM
- *Max:* 12/31/2022 12:52:00 PM

**2. Ridership Dataset**
- *Min:* 2016-01-01
- *Max:* 2022-10-31

**3. Lighting Dataset**
- *Min:* 2010-07-06
- *Max:* 2019-02-12

**4. Divvy Trips Dataset**
- *Min:* 2013-06-27 01:06:00
- *Max:* 2019-12-31 23:57:17

**5. Vacant Buildings Dataset**
- *Min:* 2008-01-18
- *Max:* 2023-10-19

### Testing and Training Datasets

Considering the specified date ranges outlined earlier, our decision was to focus our efforts on the period spanning 2016 to 2020, encompassing comprehensive data from all relevant datasets. Subsequently, we divided the extensive dataset into distinct subsets, maintaining a 3:1 ratio. Adhering to the natural chronology of a time-series dataset, our training dataset consisted of all data points recorded from 01/01/2016, to 12/31/2018, with our testing dataset incorporated data points from 01/01/2019, to 12/31/2019. The choice of this extended timeframe for the testing dataset was deliberate, aiming to ensure that our models could effectively capture hourly, daily, weekly, monthly, and yearly seasonal trends. 

Our final testing and training datasets each had the following columns:

- date (feature)
- non-violent (target)
- violent (target)
- train_rides (feature)
- bike_rides (feature)
- lighting (feature)
- vacant_buildings (feature)

**Date Ranges Visualized: **

[Date Ranges](https://github.com/evanameyer1/Machine-Learning-Crime-Prediction-Application/raw/new-main/date_ranges.png)

### Representative Samples

Considering the challenge of working with testing and training datasets for all 77 neighborhoods and the need to fit approximately 7 models per dataset, the task of managing around 1078 unique models proved to be impractical. To address this, we opted for a more manageable approach by creating 12 representative datasets. These datasets served as a basis for comparing the performance and accuracy of each model across various test cases.

Given that each adaptation involved incorporating time-series data into every model, we established three representative samples for each target column ('non-violent' and 'violent' crimes). These samples were determined based on seasonality, identified through a Dickey-Fuller test, and included metrics for 'min,' 'avg,' and 'max.' This streamlined approach allowed for a comprehensive yet more manageable evaluation of model performance.

#### 1. Dickey Fuller Test

Here, we employ the Augmented Dickey-Fuller (ADF) test from the statsmodels library to assess the stationarity of time-series data. The purpose is to identify and address any potential non-stationarity within the dataset, crucial for accurate modeling and forecasting. The function dickey_fuller_test iterates through specified target columns, conducts the ADF test for each, and returns a dictionary containing p-values indicating the stationarity of each column.:

```python
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

# Function to perform Augmented Dickey-Fuller test on specified columns
def dickey_fuller_test(df, target_cols):
    """
    :param df: DataFrame containing the time-series data
    :param target_cols: List of target columns to be tested for stationarity
    :return: Dictionary with column names as keys and corresponding p-values from ADF test as values
    """
    
    # Dictionary to store results
    output = {}

    # Loop through target columns
    for col in target_cols:
        
        # Print progress  
        print(f"{datetime.now()}: Testing {col} for stationarity...")

        try:
            # Perform ADF test
            adf_result = adfuller(df[col])
            
        except Exception as e:
            # Handle errors
            print(f"Error testing {col}. Skipping.")
            print(e)
            continue
        
        # Store p-value in output dictionary
        output[col] = adf_result[1]

    return output
```

#### 2. Representative Samples

Here we focuses on determining representative samples for each target column based on the outcomes of the Dickey-Fuller test. The function representative_sample takes two inputs - a dictionary of DataFrames (dfs) containing time-series data for different tables, and another dictionary (adf) with the results of the Dickey-Fuller test for each column. The function creates a new dictionary, representative_samples, which stores representative samples categorized by 'min,' 'max,' and 'avg' values of the Dickey-Fuller test results.

```python
# Function to determine representative samples based on Dickey-Fuller test results
def representative_sample(dfs, adf):
    """
    :param dfs: Dictionary of DataFrames containing time-series data for different tables
    :param adf: Dictionary with Dickey-Fuller test results for each column
    :return: Dictionary with representative samples for each column based on 'min,' 'max,' and 'avg' Dickey-Fuller results
    """
    
    # Dictionary to store representative samples
    representative_samples = {}

    # Loop through columns
    for col_name in adf:
        representative_samples[col_name] = {}
        
        # Store DataFrames corresponding to 'min' and 'max' Dickey-Fuller values
        representative_samples[col_name]['min'] = dfs[max(adf[col_name], key=adf[col_name].get)]
        representative_samples[col_name]['max'] = dfs[min(adf[col_name], key=adf[col_name].get)]
        
        # Calculate the average DataFrame based on 'avg' Dickey-Fuller values
        for table_name, value in adf[col_name].items():
            if 'avg' not in representative_samples[col_name].keys():
                representative_samples[col_name]['avg'] = dfs[table_name].copy()
            else:
                representative_samples[col_name]['avg'] += dfs[table_name]

        # Normalize the average DataFrame
        representative_samples[col_name]['avg'] = representative_samples[col_name]['avg'].div(len(adf[col_name]))

    return representative_samples

# Example usage
representative_samples = representative_sample(train_tables, train_adf_by_col)
```

#### 3. Visualizing Seasonality

Finally, here we execute a visualization of the seasonality within each sample, providing a visual verification of our analytical results. The function visualize_seasonality takes a list of DataFrames (dfs) as input, where each DataFrame represents a specific sample. The visualizations are created by splitting each DataFrame into intervals and plotting the averaged data for both 'non-violent' and 'violent' crimes. The resulting plots facilitate a comprehensive understanding of the temporal patterns in the dataset.

```python
import matplotlib.pyplot as plt

# Function to visualize the seasonality of each sample
def visualize_seasonality(dfs):
    """
    :param dfs: List of DataFrames representing different samples
    """
    
    # Loop through each DataFrame in the list
    for j, df in enumerate(dfs):
        
        # Define intervals for splitting the DataFrame
        intervals = [365 * 3, 52 * 3, 12 * 3]  # Number of intervals to split the DataFrame
        
        # Create subplots based on the number of intervals
        fig, axes = plt.subplots(len(intervals), 1, figsize=(30, 15))
        plt.title(f'Table {j}') 
        plt.subplots_adjust(hspace=0.4)

        # Iterate through each interval
        for count, interval in enumerate(intervals):
            # Initialize an empty DataFrame for averaging
            rows_per_interval = len(df) // interval
            average_df = df.iloc[:rows_per_interval]

            # Create a list of DataFrames, each corresponding to an interval
            interval_dfs = [df.iloc[i * rows_per_interval:(i + 1) * rows_per_interval] for i in range(1, interval)]
            
            # Average data across the interval
            for interval_df in interval_dfs:
                average_df += interval_df

            # Divide by the number of intervals to get the average
            average_df /= len(interval_dfs)

            # Create a plot for the averaged data in the current subplot
            ax = axes[count]
            ax.plot(average_df.index, average_df['non-violent'], label='Non-Violent')
            ax.plot(average_df.index, average_df['violent'], label='Violent')
            ax.legend()
            ax.grid(True)

        # Set a common x-label for all subplots
        axes[-1].set_xlabel('Date')

        # Show the entire plot
        plt.show()

# Example usage
visualize_seasonality(dfs)
```
