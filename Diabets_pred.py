# Değişkenler:
# Pregnancies: Hamile kalma sayısı.
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness : Deri kalınlığı.
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş.
# Outcome: Kişinin diyabet olup olmadığı bilgisi.

import pandas as pd
import numpy as np
from helpers.helpers import *
from helpers.eda import *
from helpers.data_prep import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load():
    data = pd.read_csv("Datasets/diabetes.csv")
    return data


df = load()
check_df(df)


# Dealing with outliers.
zero_columns = [i for i in df.columns if (df[i].min() == 0 and i not in ["Pregnancies", "Outcome"])]
zero_columns

for i in zero_columns:
    df[[i]] = df[[i]].replace(0, np.NaN)

df.isnull().sum()


def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_target(i)[i][1]

check_df(df)

num_cols = [i for i in df.columns if df[i].dtypes != "O" and df[i].nunique() > 10]

for i in num_cols:
    print(i, check_outlier(df, i))


replace_with_thresholds(df, "Insulin")
replace_with_thresholds(df, "SkinThickness")

# Feature Engineering
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

df.loc[(df["BMI"]) < 18.5, "NEW_BMI_NOM"] = "underweight"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] < 25), "NEW_BMI_NOM"] = "healthy"
df.loc[(df["BMI"] >= 25) & (df["BMI"] < 30), "NEW_BMI_NOM"] = "overweight"
df.loc[(df["BMI"]) >= 30, "NEW_BMI_NOM"] = "obese"

df.loc[(df["Glucose"]) < 70, "NEW_GLUCOSE_NOM"] = "low"
df.loc[(df["Glucose"] >= 70) & (df["Glucose"] < 100), "NEW_GLUCOSE_NOM"] = "normal"
df.loc[(df["Glucose"] >= 100) & (df["Glucose"] <= 125), "NEW_GLUCOSE_NOM"] = "hidden"
df.loc[(df["Glucose"]) > 125, "NEW_GLUCOSE_NOM"] = "high"

df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"

df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"

df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"

df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"

df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"

df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (
            (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"

df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

df["PREG_AGE"] = df["Pregnancies"] * df["Age"]

df["DiaPedFunc_Cat"] = pd.qcut(df["DiabetesPedigreeFunction"], 3, labels=["low", "middle", "up"])


def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))

df.loc[(df["BloodPressure"] < 79), "NEW_BLOODPRESSURE_CAT"] = "Normal"
df.loc[(df["BloodPressure"] > 79) & (df["BloodPressure"] < 89), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S1"
df.loc[(df["BloodPressure"] > 89) & (df["BloodPressure"] < 123), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S2"

df.head()

# Label Encoding
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    df = label_encoder(df, col)

df = rare_encoder(df, 0.01)

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols)

# Check DataFrame.
check_df(df)

# To pickle, csv.
df.to_pickle("Datasets/prepared_diabetes_df.pkl")

df = pd.read_pickle("Datasets/prepared_diabetes_df.pkl")

df.to_csv("prepared_diabetes_df.csv")

df.shape














