# import libraries that are needed
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from functools import reduce
import os
import pandas as pd
import pickle
import requests
import urllib.request


def set_working_directory():
    ''' Set current working directory into directory where script is located '''
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print("Working directory updated: " + os.getcwd())


def download_data_file(url_to_file, file_name):
    ''' Download data file (latest version) from Nord Pool '''
    latest_url = None
    version = 1
    # try to find the latest file from the server
    while True:
        complete_file_url = url_to_file
        if version == 1:
            complete_file_url += ".xls"
        else:
            complete_file_url += str(version) + ".xls"
        version += 1

        try:
            status_code = urllib.request.urlopen(complete_file_url).getcode()
            if status_code == 200:
                latest_url = complete_file_url
            else:
                break
        except:
            break
    # make sure that file was found
    if latest_url == None:
        raise Exception("Could not find file from server: " + url_to_file)
    # download the file from server and add it to input_files folder
    print("Downloading file:" + latest_url)
    r = requests.get(latest_url, allow_redirects=True)
    open(file_name, 'wb').write(r.content)


def convert_xls_into_csv(file_name, new_file_name):
    ''' Convert xls into csv file. The file is not actually xls file but html file.'''
    print("Converting the xls file into csv file")
    table = BeautifulSoup(open(file_name, 'r').read()).find('table')
    pd_list = pd.read_html(str(table))
    df = pd_list[0]
    df.to_csv(new_file_name, sep=";", index=False)


def merge_the_data(price_file, production_file, consumption_file):
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


def predict_prices(df):
    ''' Predict electricity price for each hour '''
    df = df.drop(columns=['PRICE (EUR/MWh)', 'PRODUCTION (MWh)', 'CONSUMP (MWh)'], axis = 1)
    # add missing weather variables (no weather API at least for now)
    df['MAARIANHAMINA CLOUDS (1/8)'] = 0
    df['MAARIANHAMINA TEMP (C)'] = 0
    df['MAARIANHAMINA WIND (m/s)'] = 0
    df['JYVÄSKYLÄ CLOUDS (1/8)'] = 0
    df['JYVÄSKYLÄ TEMP (C)'] = 0
    df['JYVÄSKYLÄ WIND (m/s)'] = 0
    df['KAJAANI CLOUDS (1/8)'] = 0
    df['KAJAANI TEMP (C)'] = 0
    df['KAJAANI WIND (m/s)'] = 0
    df['KUUSAMO CLOUDS (1/8)'] = 0
    df['KUUSAMO TEMP (C)'] = 0
    df['KUUSAMO WIND (m/s)'] = 0
    df['JOENSUU CLOUDS (1/8)'] = 0
    df['JOENSUU TEMP (C)'] = 0
    df['JOENSUU WIND (m/s)'] = 0
    df['OULU CLOUDS (1/8)'] = 0
    df['OULU TEMP (C)'] = 0
    df['OULU WIND (m/s)'] = 0
    df['PORI CLOUDS (1/8)'] = 0
    df['PORI TEMP (C)'] = 0
    df['PORI WIND (m/s)'] = 0
    df['KUOPIO CLOUDS (1/8)'] = 0
    df['KUOPIO TEMP (C)'] = 0
    df['KUOPIO WIND (m/s)'] = 0
    df['SODANKYLÄ CLOUDS (1/8)'] = 0
    df['SODANKYLÄ TEMP (C)'] = 0
    df['SODANKYLÄ WIND (m/s)'] = 0
    df['TURKU CLOUDS (1/8)'] = 0
    df['TURKU TEMP (C)'] = 0
    df['TURKU WIND (m/s)'] = 0
    # load the random forest regression model
    model = pickle.load(open('./models/regression_model.sav', 'rb'))
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

# run the following script when file is executed as main program
if __name__ == '__main__':
    set_working_directory()
    download_data_file(
        "https://www.nordpoolgroup.com/4ab274/globalassets/marketdata-excel-files/elspot-prices_2021_hourly_eur",
        "./input_files/2021_prices_hourly.xls")
    download_data_file(
        "https://www.nordpoolgroup.com/4aaffb/globalassets/marketdata-excel-files/consumption-per-country_2021_hourly",
        "./input_files/2021_consumption_hourly.xls")
    download_data_file(
        "https://www.nordpoolgroup.com/4ab037/globalassets/marketdata-excel-files/production-per-country_2021_hourly",
        "./input_files/2021_production_hourly.xls")
    convert_xls_into_csv("./input_files/2021_prices_hourly.xls", "./input_files/2021_prices_hourly.csv")
    convert_xls_into_csv("./input_files/2021_consumption_hourly.xls", "./input_files/2021_consumption_hourly.csv")
    convert_xls_into_csv("./input_files/2021_production_hourly.xls", "./input_files/2021_production_hourly.csv")
    df = merge_the_data(
        "./input_files/2021_prices_hourly.csv",
        "./input_files/2021_consumption_hourly.csv",
        "./input_files/2021_production_hourly.csv")
    df = more_data_cleaning(df)
    df = create_sifted_variables(df)
    predictions = predict_prices(df)
    write_predictions_to_csv(predictions)
