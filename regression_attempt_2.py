import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score
import datetime as dt
import matplotlib.pyplot as plt

#based off of Ian's work in regression_attempt_1.py
#This file includes annotations for each step

def load_data():
    '''
    load the data and select country
    returns: country_data
    '''
    data = pd.read_csv('owid-covid-data.csv')
    # print(data) #this prints out summary of csv file with 130813 rows and 65 columns
    # this is where we can change the country
    country_data = data.drop(data[data['iso_code'] != "BEL"].index)
    # print(country_data.location.unique())
    return country_data

def preprocess(country_data):
    '''
    :param country_data:
    :return: x_train, x_test, y_train, y_test
    '''

    x = country_data[['new_cases', 'date', 'reproduction_rate', 'new_tests', 'stringency_index']].copy()

    #fill in '0' for the cells where we don't have data
    x['reproduction_rate'] = x['reproduction_rate'].fillna(0)
    x['new_tests'] = x['new_tests'].fillna(0)
    x['stringency_index'] = x['stringency_index'].fillna(0)

    #convert our date string to a datetime object
    x['date'] = pd.to_datetime(x['date'])
    x['date'] = x['date'].map(dt.datetime.toordinal)

    #turn our 'new_cases' column into the label (y) and remove from the feature vector
    y = country_data[['new_cases']].copy()
    x = x.drop('new_cases', axis = 1)

    #split up the data into training and testing data (what about validation data?)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def OLS_learning(x_train, y_train):
    '''
    :return: model
    '''
    #fit our regression model to the features and labels in training data
    model = LinearRegression(n_jobs=4)
    model.fit(x_train, y_train)

    return model

def l2_learning(x_train, y_train):
    '''
    :return:
    '''
    model = Ridge(fit_intercept=0)
    model.fit(x_train, y_train)

    return model

def prediction(model, x_test, y_test):
    '''
    :return: r2 score
    '''
    #predict y and compare with actual y and return r2 score
    y_predict = model.predict(x_test)
    r2 = r2_score(y_pred=y_predict, y_true=y_test)

    return r2, y_predict

def graph_labels(x_test, y_test, OLS_predict, l2_predict):
    '''
    :return: graph comparing actual y labels with predicted labels
    '''
    print(y_test)
    print("\n")
    print(OLS_predict)
    print("\n")
    print(l2_predict)
    print(x_test['date'])

    # create data
    x = x_test['date']

    # plot lines
    plt.plot(x, y_test, label="Actual")
    plt.plot(x, OLS_predict, label="OLS")
    plt.plot(x, l2_predict, label="l2")
    plt.legend()
    plt.show()

    return
'''
Second regression attempt with some variations.
'''
def main():
    country_data = load_data()
    x_train, x_test, y_train, y_test = preprocess(country_data)
    #print(x_train, x_test, y_train, y_test)

    #my two models
    OLS_model = OLS_learning(x_train, y_train)
    ridge_model = l2_learning(x_train, y_train)

    r2_OLS, OLS_predict = prediction(OLS_model, x_test, y_test)
    #print("OLS regression r2 score:", r2_OLS)

    r2_ridge, l2_predict = prediction(ridge_model, x_test, y_test)
    #print("ridge regression r2 score:", r2_ridge)

    graph_labels(x_test, y_test, OLS_predict, l2_predict)

main()



