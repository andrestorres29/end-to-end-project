# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:40:07 2024

@author: torre
"""

import numpy as np
import pandas as pd
import streamlit as st 
import sklearn
from sklearn import preprocessing
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household] 


modelLR = joblib.load('modelLR.sav')
pipeline = joblib.load('pipeline.sav')


def main():
    st.title("California Housing Prices")
    st.markdown(
    """
    <style>
    .css-1v0mbdj {
        background-color: #009dcd;  /* Cambia el color de fondo del encabezado */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    longitude = st.number_input("Longitude",min_value=-124.35,max_value=-114.31,value="min")
    latitude = st.number_input("Latitude",min_value=32.54,max_value=41.95,value="min")
    house_med_age = st.number_input("Housing Median Age",min_value=1.0,max_value=52.0,value="min")
    total_rooms = st.number_input("Total Rooms",min_value=2.0,max_value=39320.0,value="min")
    total_bedrooms = st.number_input("Total Bedrooms",min_value=1.0,max_value=6445.0,value="min")
    population = st.number_input("Population",min_value=3.0,max_value=35682.0,value="min")
    households = st.number_input("Households",min_value=1.0,max_value=6082.0,value="min")
    median_income = st.number_input("Median Income",min_value=0.499,max_value=15.0001,value="min")
    median_house_value = st.number_input("Median House Value",min_value=14999.00,max_value=500001.0,value="min")
    ocean_proximity = st.selectbox("Ocean Proximity",["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY","ISLAND"])
    if st.button("Predict House Pricing"):
        features = [[longitude,latitude,house_med_age,total_rooms,total_bedrooms,population,households,median_income,
                     median_house_value,ocean_proximity]]
        data = {"longitude":float(longitude),"latitude":float(latitude),"housing_median_age":float(house_med_age),
                "total_rooms":float(total_rooms),"total_bedrooms":float(total_bedrooms),"population":float(population),
                "households":float(households),"median_income":float(median_income),"median_house_value":float(median_house_value),
                "ocean_proximity":ocean_proximity}
        df = pd.DataFrame([list(data.values())],columns=["longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
                                                         "population","households","median_income","median_house_value","ocean_proximity"])
        df_prepared = pipeline.transform(df)
        prediction = modelLR.predict(df_prepared)
        output = float(prediction[0])
        st.success('Prediction is {}'.format(output))
        
if __name__=='__main__': 
    main()
