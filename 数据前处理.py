
# coding: utf-8

# In[1]:


import pandas as pd

df_air_reserve = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/air_reserve.csv",low_memory=False)
df_air_store_info = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/air_store_info.csv", low_memory=False)
df_air_visit_data = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/air_visit_data.csv", low_memory=False)
df_date_info = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/date_info.csv", low_memory=False)
df_hpg_reserve = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/hpg_reserve.csv", low_memory=False)
df_hpg_store_info = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/hpg_store_info.csv", low_memory=False)
df_store_id_relation = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/store_id_relation.csv", low_memory=False)
df_sample_submission = pd.read_csv("/Users/aa370031/Dropbox/学习资料/数据分析/kaggle/Recruit Restaurant Visitor Forecasting/sample_submission.csv", low_memory=False)


# In[2]:


#air表结合
df_date_info.rename(columns={'calendar_date': 'visit_date'}, inplace=True)

df = pd.merge(df_air_visit_data, df_date_info, on='visit_date', how='left')
df_new_air = pd.merge(df, df_air_store_info, on='air_store_id', how='left')
df_new_air.sort_values(by='visit_date', inplace=True, ascending=True)


# In[35]:


#画图检验
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as tick

df_check_air = df_new_air[(df_new_air.air_store_id=='air_fab092c35776a9b1')&(df_new_air.visit_date<'2016-03-01')]

plt.plot(df_check_air.visit_date, df_check_air.visitors, marker='o')
plt.xlabel('visit_date')
plt.ylabel('visitors')
plt.legend()
plt.xticks(rotation=90, size=10)


# In[20]:


df_air_reserve.head()


# In[5]:


#hpg表结合
def trans_visit_datetime(date):
    date = date.split(' ')[0]
    return date

df_hpg_reserve['visit_date'] = ''
df_hpg_reserve.visit_date = df_hpg_reserve.visit_datetime.apply(lambda x: trans_visit_datetime(x))
df_check_hpg = pd.merge(df_hpg_reserve, df_hpg_store_info, on='hpg_store_id', how='left')
df_check_hpg = pd.merge(df_check_hpg, df_store_id_relation, on='hpg_store_id', how='left')
df_check_hpg = pd.merge(df_check_hpg, df_air_store_info, on='air_store_id', how='left')
df_check_hpg = pd.merge(df_check_hpg, df_date_info, on='visit_date', how='left')


# In[6]:


#去除饭店信息为Nan的数据
df_hpg1 = df_check_hpg[(df_check_hpg.air_store_id==df_check_hpg.air_store_id)]
df_hpg2 = df_check_hpg[(df_check_hpg.air_store_id!=df_check_hpg.air_store_id)&(df_check_hpg.hpg_area_name==df_check_hpg.hpg_area_name)]


# In[7]:


df_hpg1 = df_hpg1[['visit_date', 'reserve_datetime', 'reserve_visitors',
                 'air_store_id', 'air_genre_name', 'air_area_name', 
                  'latitude_y', 'longitude_y']]
df_hpg2 = df_hpg2[['hpg_store_id', 'visit_date', 'reserve_datetime',
                  'reserve_visitors', 'hpg_genre_name', 'hpg_area_name',
                  'latitude_x', 'longitude_x']]

df_hpg2.rename(columns={'hpg_store_id': 'air_store_id',
                        'hpg_genre_name': 'air_genre_name',
                        'hpg_area_name': 'air_area_name',
                        'latitude_x': 'latitude_y',
                        'longitude_x': 'longitude_y'}, inplace=True)


# In[8]:


df_new_hpg = pd.concat([df_hpg1, df_hpg2], axis=0, ignore_index=True)


# In[14]:


print(len(df_new_hpg))
df_new_hpg.head()


# In[15]:


print(len(df_new_air))
df_new_air.head()

