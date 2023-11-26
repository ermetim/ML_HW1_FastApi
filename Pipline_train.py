import pandas as pd
import numpy as np
import random

import re
import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

random.seed(42)
np.random.seed(42)
RANDOM = 42

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


def save_model(model, file_name='default_model.pkl'):
    """
    Сохранение модели в .pkl файл
    """
    try:
        os.mkdir('models')
    except:
        pass
    file_path = os.path.join(os.getcwd(), 'models', file_name)
    pickle.dump(model, open(file_path, 'wb'))


#     print(f'Model {name} saved successfully')


def load_model(file_name):
    """
    Загрузка сохраненной модели из .pkl файла
    """
    file_path = os.path.join(os.getcwd(), 'models', file_name)
    return pickle.load(open(file_path, 'rb'))


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

df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv')

# Удаление дубликатов
col = df_train.drop(columns=['selling_price']).columns
df_train = df_train.drop_duplicates(subset=col, keep='first').reset_index(drop=True)


df_train['mileage'] = modify_mileage(df_train['mileage'])
df_test['mileage'] = modify_mileage(df_test['mileage'])

for col in ['mileage', 'engine', 'max_power']:
    df_train[col] = drop_units(df_train[col])
    df_test[col] = drop_units(df_test[col])

df_train[['torque', 'max_torque_rpm']] = df_train['torque'].apply(modify_torque).to_list()
df_test[['torque', 'max_torque_rpm']] = df_test['torque'].apply(modify_torque).to_list()

# марка и модель авто из имени
df_train['brand'] = df_train['name'].str.split().apply(lambda x: x[0].lower())
df_train['model'] = df_train['name'].str.split().apply(lambda x: x[1].lower())

df_test['brand'] = df_test['name'].str.split().apply(lambda x: x[0].lower())
df_test['model'] = df_test['name'].str.split().apply(lambda x: x[1].lower())

# Стольбцы с nan
nan_cols = df_train.isna().sum()[df_train.isna().sum() != 0].index

# Медианы по маркам и моделям машин
median_values = df_train.groupby(by=['brand', 'model'])[nan_cols].median()
save_model(median_values, file_name='median_values_model.data')
median_values = load_model(file_name='median_values_model.data')

# заполнение nan медианами по марке и модели
def fill_na(row):
    if row.isna().sum() > 0:
        return row.fillna(median_values.loc[row['brand'], row['model']])
    return row

# Столбец с обозначением наличия nan в столбце torque
df_train['torque_isna'] = df_train['torque'].apply(lambda x: 0 if pd.notnull(x) else 1)
df_test['torque_isna'] = df_test['torque'].apply(lambda x: 0 if pd.notnull(x) else 1)

df_train = df_train.apply(fill_na, axis=1)
# df_train.isna().sum()

df_test = df_test.apply(fill_na, axis=1)
# df_test.isna().sum()

# Заполнение остатков nan медианами по столбцу
df_train[nan_cols] = df_train[nan_cols].fillna(df_train[nan_cols].median())

# квадрат года
df_train['sq_year'] = df_train['year'] ** 2
df_test['sq_year'] = df_test['year'] ** 2

# столбец с числом "лошадей" на литр объема
df_train['power/engine'] = df_train['max_power'] / df_train['engine']
df_test['power/engine'] = df_test['max_power'] / df_test['engine']

# seats в категории
df_train['seats'] = df_train['seats'].astype('object')
df_test['seats'] = df_test['seats'].astype('object')

# Удаление дубликатов
df_train = df_train.sort_values(by='year')
col = df_train.drop(columns=['selling_price']).columns
df_train = df_train.drop_duplicates(subset=col, keep='last').reset_index(drop=True)


# Разделение на признаки и целевой признак с удалением части категориальных признаков
y_train = df_train['selling_price']
X_train = df_train.drop(columns=['name', 'selling_price', 'model'])

y_test = df_test['selling_price']
X_test = df_test.drop(columns=['name', 'selling_price', 'model'])

# # StandardScaler
# num_features = (df_train.drop(columns=['selling_price', 'torque_isna'])
#                .select_dtypes(include=np.number)
#                .columns
#               )
# scaler = StandardScaler()
# scaler.fit(X_train[num_features])
# save_model(scaler, 'scaler_model.pkl')

# X_train = my_transformer(X_train, num_features, model=scaler)
# X_test = my_transformer(X_test, num_features, model_path='scaler_model.pkl')

# Список категориальных признаков для OHE
cat_features = X_train.dtypes[df_train.dtypes == 'object'].index

# OHE для категориальных признаков
ohe = OneHotEncoder(drop='first', sparse_output = False, handle_unknown='ignore')
ohe.fit(X_train[cat_features])

# Сохранение трансформатора OHE
save_model(ohe, 'ohe_model.pkl')

# OHE
X_train = my_transformer(X_train, cat_features, model=ohe)
X_test = my_transformer(X_test, cat_features, model_path='ohe_model.pkl')
# удаление трансформированных столбцов
X_train = X_train.drop(columns=cat_features)
X_test = X_test.drop(columns=cat_features)

# Логарифмирование целевого признака
y_train = np.log(y_train + 1)

# Обучение и предсказание на тесте с дефолтными гиперпараметрами
model = LinearRegression()
model.fit(X_train, y_train)
save_model(model, file_name='LinearRegression_model.pkl')
print('Pipeline finished successfully')
