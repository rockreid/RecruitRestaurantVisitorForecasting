
# coding: utf-8

# <h3>学習してみる</h3>

# In[1]:


import pandas as pd
df_preprocessed=pd.read_csv('/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/preprocessed_data.csv')


# In[2]:


# change date format to datetime
df_preprocessed['visit_date']=pd.to_datetime(df_preprocessed['visit_date'])


# In[3]:


from datetime import datetime
from datetime import timedelta

# date = df_preprocessed.loc[1, "visit_date"]
# date.year
df_preprocessed["year"] = df_preprocessed["visit_date"].apply(lambda x: x.year)
df_preprocessed["month"] = df_preprocessed["visit_date"].apply(lambda x: x.month)
df_preprocessed["week_num"] = df_preprocessed["visit_date"].apply(lambda x: x.isocalendar()[1])
df_preprocessed["pre_week_date"] = df_preprocessed["visit_date"].apply(lambda x: x - timedelta(weeks=1))
df_preprocessed["pre_week_year"] = df_preprocessed["pre_week_date"].apply(lambda x: x.year)
df_preprocessed["pre_week_month"] = df_preprocessed["pre_week_date"].apply(lambda x: x.month)
df_preprocessed["pre_week_week_num"] = df_preprocessed["pre_week_date"].apply(lambda x: x.isocalendar()[1])
df_preprocessed


# In[106]:


import numpy as np

# 取上一周的平均数，中位数，最大值，最小值
week_mean = df_preprocessed.groupby(by=["air_store_id",  "year", "week_num"]).mean()["visitors"]
week_median = df_preprocessed.groupby(by=["air_store_id",  "year", "week_num"]).median()["visitors"]
week_max = df_preprocessed.groupby(by=["air_store_id",  "year", "week_num"]).max()["visitors"]
week_min = df_preprocessed.groupby(by=["air_store_id",  "year", "week_num"]).min()["visitors"]

# df_preprocessed["pre_week_visitors_mean"] = ""
# df_preprocessed["pre_week_visitors_median"] = ""
# df_preprocessed["pre_week_visitors_max"] = ""
# df_preprocessed["pre_week_visitors_min"] = ""

# 平均数
def get_pre_week_visitors_mean(row):
    try:
        mean = week_mean[row["air_store_id"]][row["year"]][row["pre_week_week_num"]]
    except:
        mean = np.nan
    return mean

# 中位数
def get_pre_week_visitors_median(row):
    try:
        median = week_median[row["air_store_id"]][row["year"]][row["pre_week_week_num"]]
    except:
        median = np.nan
    return median

# 最大值
def get_pre_week_visitors_max(row):
    try:
        visitors_max = week_max[row["air_store_id"]][row["year"]][row["pre_week_week_num"]]
    except:
        visitors_max = np.nan
    return visitors_max

# 最小值
def get_pre_week_visitors_min(row):
    try:
        visitors_min = week_min[row["air_store_id"]][row["year"]][row["pre_week_week_num"]]
    except:
        visitors_min = np.nan
    return visitors_min

df_preprocessed["pre_week_visitors_mean"] = df_preprocessed.apply(get_pre_week_visitors_mean, axis=1)
df_preprocessed["pre_week_visitors_median"] = df_preprocessed.apply(get_pre_week_visitors_median, axis=1)
df_preprocessed["pre_week_visitors_max"] = df_preprocessed.apply(get_pre_week_visitors_max, axis=1)
df_preprocessed["pre_week_visitors_min"] = df_preprocessed.apply(get_pre_week_visitors_min, axis=1)


# In[116]:


df_preprocessed.dropna(how="any", inplace=True, axis="rows")


# In[117]:


#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# In[118]:


store_id_list = ["air_9b6af3db40da4ae2","air_20add8092c9bb51d","air_1033310359ceeac1",
                 "air_640cf4835f0d9ba3","air_8093d0b565e9dbdf","air_4092cfbd95a3ac1b",
                 "air_1408dd53f31a8a65","air_dea0655f96947922","air_12c4fb7a423df20d",
                 "air_a083834e7ffe187e","air_6b15edd1b4fbb96a","air_43b65e4b05bff2d3",
                 "air_707d4b6328f2c2df","air_ca6ae8d49a2f1eaf","air_de803f7e324936b8",
                 "air_fe22ef5a9cbef123","air_28064154614b2e6c","air_e053c561f32acc28",
                 "air_3c05c8f26c611eb9","air_b23d0f519291247d"]

def get_top20_store(input_id):
    if input_id in store_id_list:
        flg = 1
    else:
        flg = 0
    return flg


# In[119]:


df_preprocessed["top20_store_flg"] = df_preprocessed.air_store_id.apply(lambda x: get_top20_store(x))


# In[120]:


def get_days_from_start(input_days, df):
    days_from_start = (input_days - df.visit_date.min()).days
    return days_from_start


# In[130]:


import datetime

df_learn_test = pd.DataFrame()
df_validate_test = pd.DataFrame()

col_s=datetime.datetime(2016, 7, 1)
last_col_s=datetime.datetime(2016,11,25)

while(col_s<=last_col_s):
    end_date = col_s + datetime.timedelta(days=10)
    col_e=col_s + datetime.timedelta(days=27)
    val_s=col_s + datetime.timedelta(days=28)
    val_e=col_s + datetime.timedelta(days=34)
    
    df_learn_test = df_learn_test.append(df_preprocessed[(col_s<=df_preprocessed['visit_date']) & (df_preprocessed['visit_date']<=col_e)])
    df_validate_test = df_validate_test.append(df_preprocessed[(val_s<=df_preprocessed['visit_date']) & (df_preprocessed['visit_date']<=val_e)])
    col_s=col_s + datetime.timedelta(days=7)
    
df_learn_test.reset_index(inplace=True, drop=True)
df_validate_test.reset_index(inplace=True, drop=True)
df_learn_test["days_from_start"] = df_learn_test["visit_date"].apply(lambda x: get_days_from_start(x, df=df_learn_test))
df_validate_test["days_from_start"] = df_validate_test["visit_date"].apply(lambda x: get_days_from_start(x, df=df_learn_test))


# In[200]:


df_learn_test = df_learn_test[df_learn_test["top20_store_flg"]==1]
df_validate_test = df_validate_test[df_validate_test["top20_store_flg"]==1]


# In[208]:


import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

X_train = df_learn_test.drop(['air_store_id', 'visitors', 'visit_date', "month", "year", "pre_week_date", 
                            "top20_store_flg"], axis="columns") 
print("投入模型的特征为: {}".format(X_train.columns))
y_train=df_learn_test['visitors']

X_test = df_validate_test.drop(['air_store_id', 'visitors', 'visit_date', "month", "year", "pre_week_date", 
                             "top20_store_flg"], axis="columns")
y_test=df_validate_test['visitors']

reg = ExtraTreeRegressor().fit(X_train, y_train)
rms = sqrt(mean_squared_error(y_test, reg.predict(X_test)))
# print(r2_score(reg.predict(X_test), y_test))
print(rms)
print("予測値：{}".format(reg.predict(X_test)))
print("実績：{}".format(y_test.values))


# plt.title("Regression curve")
# plt.plot()
# plt.xlabel('date')
# plt.ylabel('visitors')


# In[205]:


# %matplotlib inline
# # ax=df_validate_test[(df_validate_test['air_store_id']=='air_06f95ac5c33aca10')&((df_validate_test["month"]==8)|(df_preprocessed["month"]==9))].plot(x='visit_date', y='visitors', figsize=(10,5), grid=True)
# ax = df_validate_test[(df_validate_test["month"]==8)|(df_validate_test["month"]==9)].plot(x='visit_date', y='visitors', figsize=(10,5), grid=True)
# ax.plot()


# In[105]:


# df_preprocessed.air_store_id.unique()

