# import libraries that are needed
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from functools import reduce
from itertools import chain
import json
import numpy as np
import os
import pandas as pd
import pickle
from pandas.io.stata import invalid_name_doc
import requests



def set_working_directory():
    ''' Set current working directory into directory where script is located '''
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print("Working directory updated: " + os.getcwd())


def download_data_file(url_to_file, file_name):
    ''' Download data file '''
    print("Downloading file:" + url_to_file)
    r = requests.get(url_to_file, allow_redirects=True)
    open(file_name, 'wb').write(r.content)


def convert_xls_into_csv(file_name, new_file_name):
    ''' Convert xls into csv file. The file is not actually xls file but html file.'''
    print("Converting the xls file into csv file")
    table = BeautifulSoup(open(file_name, 'r').read()).find('table')
    pd_list = pd.read_html(str(table))
    df = pd_list[0]
    df.to_csv(new_file_name, sep=";", index=False)

def convert_weather_json_into_csv(filename_json, filename_csv):
    with open(filename_json) as json_file:
        weather_data_dict = json.load(json_file)
    # fetching the needed information from hourly weather values
    hourly_weathers = weather_data_dict["forecastValues"]
    years = [int(one_datapoint["localtime"][0:4]) for one_datapoint in hourly_weathers]
    months = [int(one_datapoint["localtime"][4:6]) for one_datapoint in hourly_weathers]
    dates = [int(one_datapoint["localtime"][6:8]) for one_datapoint in hourly_weathers]
    hours = [int(one_datapoint["localtime"][9:11]) for one_datapoint in hourly_weathers]
    temperatures = [one_datapoint["Temperature"] for one_datapoint in hourly_weathers]
    wind_speeds = [one_data_point["WindSpeedMS"] for one_data_point in hourly_weathers]

    # measurement station city
    measurement_city = str(hourly_weathers[0]["name"]).upper()


    # df for forecast on one weather station
    weather_station_np = np.array([years, months, dates, hours, temperatures, wind_speeds]).transpose()
    weather_station_df = pd.DataFrame(weather_station_np, columns=["Year", "Month", "Day", "Hour",'{} TEMP (C)'.format(measurement_city), '{} WIND (m/s)'.format(measurement_city)])
    weather_station_df = weather_station_df.astype("int32", errors="ignore")
    weather_station_df.to_csv(filename_csv, sep=";", index=False)


def merge_electricity_data(price_file, production_file, consumption_file):
    ''' Function to merge csv-files together. Returns pandas Dataframe '''
    # clean price data before merging process
    df_price = pd.read_csv(price_file, sep=";", skiprows=2)
    df_price = df_price.rename(columns={df_price.columns[0]: "Date"})
    df_price = df_price[['Date', 'Hours', 'FI']]
    df_price['Hours'] = df_price['Hours'].map(
        lambda hours_str: int(hours_str[0:2]))
    df_price = df_price.rename(columns={'Hours': 'Hour'})
    df_price = df_price.rename(columns={'FI': 'PRICE (EUR/MWh)'})
    # the csv conversion fails with price data (does not understand comma)
    # to fix this problem we need to divide the values with 100
    df_price['PRICE (EUR/MWh)'] = df_price['PRICE (EUR/MWh)'].apply(lambda x: x/100)
    # clean production data before merging process
    df_production = pd.read_csv(production_file, sep=";", skiprows=2)
    df_production = df_production.rename(
        columns={df_production.columns[0]: "Date"})
    df_production = df_production[['Date', 'Hours', 'FI']]
    df_production['Hours'] = df_production['Hours'].map(
        lambda hours_str: int(hours_str[0:2]))
    df_production = df_production.rename(
        columns={'Hours': 'Hour', 'FI': 'PRODUCTION (MWh)'})
    # clean consumption data before merging process
    df_consumption = pd.read_csv(consumption_file, sep=";", skiprows=2)
    df_consumption = df_consumption.rename(
        columns={df_consumption.columns[0]: "Date"})
    df_consumption = df_consumption[['Date', 'Hours', 'FI']]
    df_consumption['Hours'] = df_consumption['Hours'].map(
        lambda hours_str: int(hours_str[0:2]))
    df_consumption = df_consumption.rename(
        columns={'Hours': 'Hour', 'FI': 'CONSUMP (MWh)'})
    # do the actual merge
    data_frames = [
        df_production,
        df_consumption,
        df_price
    ]
    df = reduce(lambda left, right: pd.merge(
        left, right, on=['Date', 'Hour'], how='outer'), data_frames)
    return df

def merge_weather_data(weather_data_csv_uris, filename):
    weather_stations_df = []

    # read station specific weather csvs and add them to a list
    for csv_weather_uri in weather_data_csv_uris:
        weather_df = pd.read_csv(csv_weather_uri, sep=";")
        weather_stations_df.append(weather_df)

    # combine everything in a single df 
    combined_df = reduce(lambda left, right: pd.merge(
        left, right, on=["Year", "Month", "Day", "Hour"], how='outer'), weather_stations_df)

    combined_df.to_csv("weather_data.csv")
    
    return combined_df


def more_data_cleaning(df):
    ''' Does more data cleaning for the dataframe resulting from the merge '''
    # remove data that is older than three weeks (it wont be needed)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    today = datetime.now()
    delta = timedelta(weeks=3)
    start = today - delta
    df = df[df['Date'] >= start]
    # create weekday variable
    df['Weekday'] = df.apply(lambda row: row['Date'].weekday(), axis=1)
    # convert date to multiple variables
    df['Month'] = df.apply(lambda row: row['Date'].month, axis=1)
    df['Day'] = df.apply(lambda row: row['Date'].day, axis=1)
    df = df.drop(['Date'], errors='ignore', axis=1)
    # drop rows that have NA variables (probably the last rows)
    df = df.dropna()
    # df.to_csv("./debugging/test.csv", sep=";", index=False)
    # set the order of variables
    df = df.reindex(['Month', 'Day', 'Hour', 'Weekday', 'PRODUCTION (MWh)', 'CONSUMP (MWh)', 'PRICE (EUR/MWh)'], axis=1)
    return df


def create_rows_for_n_days(df, n):
    last_row = df.iloc[-1]
    last_day = last_row['Day']
    last_month = last_row['Month']
    # remove rows with last day and month combinations
    df = df.drop(df[(df['Day'] == last_day) & (df['Month'] == last_month)].index)
    # start creating new rows up to n days ahead
    for row in range(n):
        for hour in range(24):
            base_date = datetime(datetime.now().year, int(last_month), int(last_day))
            delta = timedelta(hours=(row*24 + hour))
            next_day = base_date + delta
            df.loc[len(df)] = [next_day.month, next_day.day, next_day.hour, next_day.weekday(), 0, 0, 0]
    df.to_csv("./debugging/test.csv", sep=";", index=False)
    return df


def create_columns_with_sift(df, column_name, arr_of_sifts):
    for sift in arr_of_sifts:
        new_column_name = column_name + ' (day - ' + str(sift) + ')'
        df[new_column_name] = df[column_name].shift(sift*24)


def create_sifted_variables(df):
    ''' Creates sifted variables for price production and consumption data '''
    # create rows up to seven days into future
    df = create_rows_for_n_days(df, 7)
    # create columns with sifted values
    sift_days = [11, 10, 9, 8, 7]
    create_columns_with_sift(df, 'PRODUCTION (MWh)', sift_days)
    create_columns_with_sift(df, 'CONSUMP (MWh)', sift_days)
    create_columns_with_sift(df, 'PRICE (EUR/MWh)', sift_days)
    # most likely rows at the beginning are missing values after shift
    df = df.dropna()
    return df


def predict_prices(df, model_file):
    ''' Predict electricity price for each hour '''
    df = df.drop(columns=['PRICE (EUR/MWh)', 'PRODUCTION (MWh)', 'CONSUMP (MWh)'], axis = 1)
    # load the random forest regression model
    model = pickle.load(open(model_file, 'rb'))
    # predict values
    predictions = model.predict(df)
    df['PREDICTION (EUR/MWh)'] = predictions
    return df


def write_predictions_to_csv(predictions):
    ''' Write results to csv '''
    df_result = predictions[['Month', 'Day', 'Hour', 'PREDICTION (EUR/MWh)']]
    df_result['Year'] = datetime.now().year
    df_result = df_result.reindex(['Year', 'Month', 'Day', 'Hour', 'PREDICTION (EUR/MWh)'], axis=1)
    df_result.to_csv("./output_files/results.csv", sep=";", index=False)
    print("PREDICTIONS DONE WITHOUT ERRORS!")

def weather_urls_from_config(file_path):
    weather_urls_dict = {}
    with open(file_path) as file:
        for line in file:
            line_splitted = line.split(",")
            weather_urls_dict[line_splitted[0].strip()] = line_splitted[1].strip()
    return weather_urls_dict

# run the following script when file is executed as main program
if __name__ == '__main__':
    set_working_directory()

    #electricity data
    download_data_file(
        "https://www.nordpoolgroup.com/globalassets/marketdata-excel-files/elspot-prices_2021_hourly_eur.xls",
        "./input_files/2021_prices_hourly.xls")
    download_data_file(
        "https://www.nordpoolgroup.com/globalassets/marketdata-excel-files/consumption-per-country_2021_hourly.xls",
        "./input_files/2021_consumption_hourly.xls")
    download_data_file(
        "https://www.nordpoolgroup.com/globalassets/marketdata-excel-files/production-per-country_2021_hourly.xls",
        "./input_files/2021_production_hourly.xls")

    convert_xls_into_csv("./input_files/2021_prices_hourly.xls", "./input_files/2021_prices_hourly.csv")
    convert_xls_into_csv("./input_files/2021_consumption_hourly.xls", "./input_files/2021_consumption_hourly.csv")
    convert_xls_into_csv("./input_files/2021_production_hourly.xls", "./input_files/2021_production_hourly.csv")

    electricity_df = merge_electricity_data(
        "./input_files/2021_prices_hourly.csv",
        "./input_files/2021_consumption_hourly.csv",
        "./input_files/2021_production_hourly.csv")
    electricity_df = more_data_cleaning(electricity_df)
    electricity_df = create_sifted_variables(electricity_df)

    electricity_df.to_csv("test.csv", sep=";")

    # weather forecast data
    weather_urls_dict = weather_urls_from_config("./config/weather_urls.txt")

    weather_csv_file_uris = []
    for key in weather_urls_dict:
        filename_json = './input_files/{}_weather_data.json'.format(key)
        filename_csv = './input_files/{}_weather_data.csv'.format(key)
        download_data_file(weather_urls_dict[key], filename_json)
        convert_weather_json_into_csv(filename_json, filename_csv)
        weather_csv_file_uris.append(filename_csv)
    
    weather_df = merge_weather_data(weather_csv_file_uris, "all_stations_weather_data.csv")

    # combine electricity and weather dataframes
    combined_df = pd.merge(electricity_df, weather_df.drop(["Year"], axis=1), on=["Month", 'Day', 'Hour'])
    combined_df = combined_df.dropna()
    
    # changing the order of the columns ß
    columns = combined_df.columns
    new_order = list(chain(columns[0:7],columns[22:],columns[7:22]))

    combined_df = combined_df[new_order]
    combined_df.to_csv("test.csv")




    # generating predictions
    predictions = predict_prices(combined_df, "./models/regression_model_wo_clouds.sav")
    write_predictions_to_csv(predictions)
