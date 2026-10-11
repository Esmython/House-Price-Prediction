import warnings

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import graphviz
import pydotplus
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    cross_val_score,
    validation_curve,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_graphviz,
    export_text,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    VotingClassifier,
)

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from skompiler import skompile

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# train ve test dosyalarını dataframe'e aktarma ve birleştirme
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_ = pd.concat([df_train, df_test], ignore_index=True)
df = df_.copy()
df.head()
df.info()
df.describe().T
# Tekrarlayan veri kontrolü
df_= df_.drop(["Id"], axis=1)
df_.duplicated().sum()

# Sayısal ve kategorik değişkenlerin tespiti
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

df["MSSubClass"] = df["MSSubClass"].astype(object)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
"MSSubClass" in  cat_cols
num_cols
cat_cols

# Veritiplerinin ve gerçekdışı verinin düzeltilmesi
df["GarageYrBlt"].sort_values(ascending=False).head(10) # 1 tane 2207 yılı verisini düzeltmeliyiz.
df.loc[df["GarageYrBlt"] > 2010, "GarageYrBlt"] = 2007

datecolumns = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
for col in datecolumns:
    df[col] = pd.to_datetime(df[col], format="%Y")
df["YearBuilt"].max()
df["YearRemodAdd"].max()
df["GarageYrBlt"].max()
today = pd.to_datetime("01.01.2011", format="%d.%m.%Y")
df["BltAge"] = today.year - df["YearBuilt"].dt.year
df["RemodAge"] = today.year - df["YearRemodAdd"].dt.year
df["GarageAge"] = today.year - df["GarageYrBlt"].dt.year

df = df.drop(datecolumns, axis=1)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

addnum = ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "Fireplaces", "GarageCars"]
for col in addnum:
    cat_cols.remove(col)
    num_cols.append(col)

# Sayısal ve kategorik değişkenlerin veri içindeki dağılımını gözlemleme
print(f"Kategorik değişken sayısı: {len(cat_cols)}\n"
      f"Sayısal değişken sayısı: {len(num_cols)}\n"
      f"Kategorik ama kardinal değişken sayısı: {len(cat_but_car)}")

# Kategorik değişkenler ile hedef değişken incelemesini
for col in cat_cols:
    print(df.groupby(col)["SalePrice"].mean().sort_values(ascending=False))

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Aykırı gözlem tespiti
df.describe().T
df[num_cols].describe(percentiles=[0.05, 0.25, 0.40,
                                   0.50, 0.75,
                                   0.90, 0.95, 0.99]).T
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
# 0.25-0.75 YearRemodAdd hariç hepsinde aykırı değer var
# 0.05-0.95
outlier_true = []
for col in num_cols:
    if check_outlier(df,col):
        outlier_true.append(col)
outlier_true
len(outlier_true)

# Eksik gözlem tespiti
pd.set_option('display.max_rows', None)
df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Eksik ve aykırı değerler ile  ilgili düzenlemeler
bsmt_cols = ["BsmtQual","BsmtCond" ,"BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
df[bsmt_cols] = df[bsmt_cols].fillna("NoBsmt")

garage_cols = ["GarageFinish", "GarageQual", "GarageCond"]
df[garage_cols] = df[garage_cols].fillna("NoGar")

missing_values_table(df)

na_columns = [col for col in df.columns if df[col].isnull().sum() > 0 and col != "SalePrice"]

deletecols = [col for col in na_columns if (df[col].isnull().sum() / df.shape[0] * 100) > 40]
df = df.drop(deletecols, axis=1)

len(deletecols)
len(df.columns)
num_cols = [col for col in num_cols if col in df.columns]
cat_cols = [col for col in cat_cols if col in df.columns]

num_cols = [col for col in num_cols if col != "SalePrice"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

missing_values_table(df)
df.isnull().sum()

df.describe().T

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)
for col in num_cols:
    print(check_outlier(df,col))

# Rare Encoder  ile nadir görülen değerlerin gruplanarak etiketlenmesi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}).sort_values("COUNT",ascending=False), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns
                    if temp_df[col].dtypes == "O"
                    and col != "Id"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])

    return temp_df

new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "SalePrice", cat_cols)

df = rare_encoder(df, 0.01)
rare_analyser(df, "SalePrice", cat_cols)

# Yeni değişkenlerin oluşturulması
df["CompRate"] = (df["BsmtFinSF1"] + df["BsmtFinSF2"]) / (df["1stFlrSF"] + df["2ndFlrSF"])
df["TotalPorch"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
df["CompRate"].head()
df["TotalPorch"].head()
num_cols = num_cols + ["CompRate", "TotalPorch"]

# Gereksiz Kolonların çıkarılması
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
               (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
df = df.drop(useless_cols, axis=1)

for col in df.columns:
    print(f"{col}:", df[col].nunique())
df["3SsnPorch"].head()
useless_cols2 = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(useless_cols2, axis=1)

cat_cols = [col for col in cat_cols if col in df.columns]
num_cols = [col for col in num_cols if col in df.columns]

# Encoding işlemleri
df["CentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0})

# --- Lot / arazi ---
df["LotShape"]     = df["LotShape"].map({"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1})
df["LandContour"]  = df["LandContour"].map({"Lvl": 4, "Bnk": 3, "HLS": 2, "Low": 1})
df["LandSlope"]    = df["LandSlope"].map({"Gtl": 3, "Mod": 2, "Sev": 1})

# --- Dış cephe ---
df["ExterQual"]    = df["ExterQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
df["ExterCond"]    = df["ExterCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})

# --- Bodrum (NA = bodrum yok = 1) ---
df["BsmtQual"]     = df["BsmtQual"].map({"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "NoBsmt": 1})
df["BsmtCond"]     = df["BsmtCond"].map({"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "NoBsmt": 1})
df["BsmtExposure"] = df["BsmtExposure"].map({"Gd": 5, "Av": 4, "Mn": 3, "No": 2, "NoBsmt": 1})
df["BsmtFinType1"] = df["BsmtFinType1"].map({"GLQ": 7, "ALQ": 6, "BLQ": 5, "Rec": 4,
                                             "LwQ": 3, "Unf": 2, "NoBsmt": 1})
df["BsmtFinType2"] = df["BsmtFinType2"].map({"GLQ": 7, "ALQ": 6, "BLQ": 5, "Rec": 4,
                                             "LwQ": 3, "Unf": 2, "NoBsmt": 1})

# --- Isıtma / mutfak / işlevsellik ---
df["HeatingQC"]    = df["HeatingQC"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
df["KitchenQual"]  = df["KitchenQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1})
df["Functional"]   = df["Functional"].map({"Typ": 8, "Min1": 7, "Min2": 6, "Mod": 5,
                                           "Maj1": 4, "Maj2": 3, "Sev": 2, "Sal": 1})
# --- Garaj (NA = garaj yok = 1) ---
df["GarageFinish"] = df["GarageFinish"].map({"Fin": 4, "RFn": 3, "Unf": 2, "NoGar": 1})
df["GarageQual"]   = df["GarageQual"].map({"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "NoGar": 1})
df["GarageCond"]   = df["GarageCond"].map({"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "NoGar": 1})
df["PavedDrive"]   = df["PavedDrive"].map({"Y": 3, "P": 2, "N": 1})

def target_group_encode(dataframe, col, target="SalePrice", q=5, fallback=3):
    means = dataframe.groupby(col)[target].mean().dropna().sort_values()
    groups = pd.qcut(means, q, labels=range(1, q + 1)).astype(int)
    encoded = dataframe[col].map(groups).fillna(fallback).astype(int)
    return encoded, groups

for col in ["Neighborhood", "Exterior1st", "Exterior2nd"]:
    df[col + "_GRP"], mapping = target_group_encode(df, col)
    print(f"\n{col} -> {col}_GRP")
    print(mapping.sort_values())

for col in ["Neighborhood_GRP", "Exterior1st_GRP", "Exterior2nd_GRP"]:
    print(df.groupby(col)["SalePrice"].agg(["mean", "count"]), "\n")

df["Exterior1st_GRP"].head()
df.info()

df["MSSubClass"].nunique()
df["MSSubClass"] = df["MSSubClass"].astype(object)

for col in ["MSSubClass"]:
    df[col + "_GRP"], mapping = target_group_encode(df, col)
    print(f"\n{col} -> {col}_GRP")
    print(mapping.sort_values())

print(df.groupby("MSSubClass_GRP")["SalePrice"].agg(["mean", "count"]))

df = df.drop(["Neighborhood", "Exterior1st", "Exterior2nd", "MSSubClass"], axis=1)

#4 değişkeni num_cols a ekle

cat_cols = [col for col in cat_cols if col in df.columns]
num_cols = [col for col in num_cols if col in df.columns]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df.info()
ohe_cols = [col for col in cat_cols if 10 >= df[col].nunique() > 1]
df = one_hot_encoder(df, ohe_cols)

cat_cols = [col for col in cat_cols if col in df.columns]
num_cols = [col for col in num_cols if col in df.columns]
missing_values_table(df)

len(df.columns)

# train ve test verisini geri ayırma
df["Id"] = df["Id"].astype(object)
train_df = df[df["SalePrice"].notnull()].copy()
test_df = df[df["SalePrice"].isnull()].copy()

# Modelleme
y = train_df["SalePrice"]
X = train_df.drop(["Id", "SalePrice"], axis=1)

models = [('LR', LinearRegression()),
          ('CART', DecisionTreeRegressor(random_state=17)),
          ('RF', RandomForestRegressor(random_state=17)),
          ('GBM', GradientBoostingRegressor(random_state=17)),
          ('XGBoost', XGBRegressor(random_state=17)),
          ('LightGBM', LGBMRegressor(random_state=17, verbose=-1)),
          ('CatBoost', CatBoostRegressor(random_state=17, verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5,
                                            scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 2)} ({name})")


df["SalePrice"].mean()
#180921.19
 #log dönüşümü
y_log = np.log1p(y)

rmse_log = np.mean(np.sqrt(-cross_val_score(LGBMRegressor(random_state=17, verbose=-1),
                                            X, y_log, cv=5,
                                            scoring="neg_mean_squared_error")))
print("Log RMSE:", rmse_log)
# 0.1332
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=17)
model = LGBMRegressor(random_state=17, verbose=-1).fit(X_train, y_train)

y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)
print("Gerçek ölçekte RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
# 25765.33

# Hiperparametre optimizasyonu
catboost_model = CatBoostRegressor(random_state=17, verbose=False)

catboost_params = {"iterations": [500, 1000, 1500],      # n_estimators karşılığı
                   "depth": [4, 6, 8, 10],               # max_depth karşılığı
                   "min_data_in_leaf": [1, 5, 10],       # min_samples_split'e en yakın
                   "learning_rate": [0.03, 0.05, 0.1]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
catboost_best_grid.best_params_

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X_train, y_train)

y_pred_real = np.expm1(catboost_final.predict(X_test))
y_true_real = np.expm1(y_test)
print(f"Gerçek ölçek RMSE: {np.sqrt(mean_squared_error(y_true_real, y_pred_real)):.2f}")
#22898.06
#23497.00
#22868.15

# Adım4
def plot_importance(model, features, num=20, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(catboost_final, X, save=True)

#Tahminleme
X_kaggle = test_df.drop(["Id", "SalePrice"], axis=1)
y_kaggle = np.expm1(catboost_final.predict(X_kaggle))

kg = pd.DataFrame()
kg["Id"] = test_df["Id"]
kg["SalePrice"] = y_kaggle
kg.head()
kg["Id"] = kg["Id"].astype(int)

kg.to_csv("houseprice.csv", index=False)
kg.info()