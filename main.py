# Imports
import pandas as pd
import numpy as np

import re
import os
import pickle

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from io import BytesIO
import json

def modify_mileage(series, koef=0.75):
    """
    Перевод единиц 'km/kg' в 'kmpl'
    """
    series = series.astype('str').apply(
        lambda x: x
        if (x is np.nan
            or 'km/kg' not in x.split()
            or len(x.split()) < 2
            )
        else str(round(float(x.split()[0]) * koef)) + ' kmpl'
    )
    return series


def drop_units(series):
    """
    Удаление единиц изменения
    15 km -> 15
    """
    series = series.astype('str').apply(
        lambda x: x.split()[0]
        if len(x.split()) > 1
        else np.nan
    ).astype('float')
    return series


def modify_torque(row):
    """
    Разбиение столбца torque на torque и max_torque_rpm
    91Nm@ 4250rpm -> 91 и 4250
    """
    if row is np.nan:
        return (np.nan, np.nan)
    numbers = re.findall(r'([\d.,]+)', row.lower())
    units = re.findall(r'(nm|rpm|kgm)', row.lower())
    if 'kgm' in units:
        torque = round(float(numbers[0]) * 9.81, 2)
    else:
        torque = float(numbers[0])
    max_torque_rpm = float(numbers[-1].replace(',', ''))

    return torque, max_torque_rpm


def load_model(file_name):
    """
    Загрузка сохраненной модели из .pkl файла
    """
    # file_path = os.path.join(os.getcwd(), 'models', file_name)
    return pickle.load(open(file_name, 'rb'))


def my_transformer(df, features_to_transform, model=None, model_path='transformer_model.pkl'):
    """
    Трансформирование признаков
    Модель трансформатора подается на вход или загружается из .pkl файла
    на вход подается полностью датафрейм с указанием признаков для трансформации
    на выходе трансформированный датафрейм
    """
    if model is None:
        model = load_model(model_path)
    values = model.transform(df[features_to_transform])
    labels = model.get_feature_names_out()
    df[labels] = values
    #     df = df.drop(columns=cat_features)
    return df


def preprocessing(df):
    df_test = df.copy()
    # Приведение значений единиц измерения к одному типу
    df_test['mileage'] = modify_mileage(df_test['mileage'])

    # Удаление единиц измерения
    for col in ['mileage', 'engine', 'max_power']:
        df_test[col] = drop_units(df_test[col])

    # Разделение столбца "torque"
    df_test[['torque', 'max_torque_rpm']] = df_test['torque'].apply(modify_torque).to_list()

    # марка и модель авто из имени
    df_test['brand'] = df_test['name'].str.split().apply(lambda x: x[0].lower())
    df_test['model'] = df_test['name'].str.split().apply(lambda x: x[1].lower())

    # Столбец с обозначением наличия nan в столбце torque
    df_test['torque_isna'] = df_test['torque'].apply(lambda x: 0 if pd.notnull(x) else 1)

    # Медианы по маркам и моделям машин
    median_values = load_model(file_name='models/median_values_model.data')

    # заполнение nan медианами по марке и модели
    def fill_na(row):
        if row.isna().sum() > 0:
            return row.fillna(median_values.loc[row['brand'], row['model']])
        return row

    if df_test.isna().sum().sum() > 0:
        # Заполнение столбцов с nan
        df_test = df_test.apply(fill_na, axis=1)

        # Стольбцы с nan
        nan_cols = df_test.isna().sum()[df_test.isna().sum() != 0].index

        # Заполнение остатков nan медианами по столбцу
        df_test[nan_cols] = df_test[nan_cols].fillna(df_test[nan_cols].median())

    # квадрат года
    df_test['sq_year'] = df_test['year'] ** 2

    # столбец с числом "лошадей" на литр объема
    df_test['power/engine'] = df_test['max_power'] / df_test['engine']

    # seats в категории
    df_test['seats'] = df_test['seats'].astype('object')

    # Удаление ненужных признаков
    X_test = df_test.drop(columns=['name']) #, 'model'])

    # Список категориальных признаков для OHE
    cat_features = X_test.dtypes[X_test.dtypes == 'object'].index

    # OHE
    X_test = my_transformer(X_test, cat_features, model_path='models/ohe_model.pkl')
    # удаление трансформированных столбцов
    X_test = X_test.drop(columns=cat_features)
    return X_test

app = FastAPI()
"""uvicorn main:app --reload
(запуск веб-сервера с указанием начального скрипта main.py и объекта
FastAPI который создали инструкцией app = FastAPI())
Обращение http://127.0.0.1:8000
"""

@app.get("/")
def root():
    return {"message": "The service is live"}


@app.post('/predict_items_csv', summary="Predict by csv")
def upload_csv(file: UploadFile):
    content = file.file.read() #считываем байтовое содержимое
    buffer = BytesIO(content) #создаем буфер типа BytesIO
    df_test = pd.read_csv(buffer) #, index_col=0)
    # buffer.close()
    # file.close()  # закрывается именно сам файл
    # # на самом деле можно не вызывать .close(), тк питон сам освобождает память при уничтожении объекта
    # # но это просто хорошая практика, так гарантируется корректное освобождение ресурсов

    X_test = preprocessing(df_test)
    model = load_model(file_name='models/LinearRegression_model.pkl')
    # prediction = model.predict(X_test)
    prediction = np.round(np.exp(model.predict(X_test)) - 1)
    df_test['prediction'] = prediction

    # try:
    #     os.mkdir('results')
    # except:
    #     pass
    # file_path = os.path.join(os.getcwd(), 'results', 'prediction.csv')
    df_test.to_csv('prediction.csv')
    response = FileResponse(path='prediction.csv', media_type='text/csv', filename='prediction.csv')
    return response

# uvicorn main:app --host 0.0.0.0 --port 8000 # For render.com

