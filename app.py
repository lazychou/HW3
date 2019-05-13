import numpy as np
import pandas as pd
import random as rd
import datetime 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import time
from itertools import product
#import sys
#import gc
import pickle
from xgboost import XGBRegressor



def preprocessing():

    ts = time.time()
    ## preprocessing
    # Import all of them 
    train = pd.read_csv("./Data/sales_train_v2.csv")
    cats = pd.read_csv("./Data/item_categories.csv")
    items = pd.read_csv("./Data/items.csv")
    shops = pd.read_csv("./Data/shops.csv")
    test = pd.read_csv("./Data/test.csv")

    # outliers
    train = train[train.item_cnt_day<1001]
    train = train[train.item_price<100000]

    # one item_price is -1, fill it with median.
    median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
    train.loc[train.item_price<0, 'item_price'] = median

    # one shope has two shpo names 
    # Якутск Орджоникидзе, 56
    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11

    # shop_name -> shop_city and shpo_name
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id','city_code']]

    # cat -> type and subtype
    cats['split'] = cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].map(lambda x: x[0].strip())
    cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
    # if subtype is nan then type
    cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
    cats = cats[['item_category_id','type_code', 'subtype_code']]
    # remove 'item_name' -> 'item_id' and 'item_category_id'
    items.drop(['item_name'], axis=1, inplace=True)

    # Monthly sales
    matrix = []
    cols = ['date_block_num','shop_id','item_id']
    #'date_block_num' -> 0,...,33
    for i in range(34):
        sales = train[train.date_block_num == i]
        matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols,inplace=True)
    #compute 'revenue'
    train['revenue'] = train['item_price'] *  train['item_cnt_day']
    #day -> month 
    group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=cols, how='left')
    # after merge some value is NaN, fill by zero
    matrix['item_cnt_month'] = (matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16))


    # Test data
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)

    matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
    matrix.fillna(0, inplace=True) # 34 month

    matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items, on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
    matrix['city_code'] = matrix['city_code'].astype(np.int8)
    matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
    matrix['type_code'] = matrix['type_code'].astype(np.int8)
    matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
    #平均出現NaN 不知哪有問題
    print(1)


    def lag_feature(df, lags, col):
        #add new col
        tmp = df[['date_block_num','shop_id','item_id',col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        return df

    matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')

    # Mean encoded features
    # date_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
    matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
    matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
    # date_item_avg_item_cnt_lag_1,2,3,6,12
    group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_item_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
    matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
    matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
    # date_shop_avg_item_cnt_lag_1,2,3,6,12
    group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_shop_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
    matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
    matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
    #date_cat_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_cat_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
    matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
    matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
    #date_shop_cat_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_cat_avg_item_cnt']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
    matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
    matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)
    # date_shop_type_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_type_avg_item_cnt']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
    matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
    matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)
    # date_shop_subtype_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_subtype_avg_item_cnt']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
    matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
    matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)
    # date_city_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_city_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
    matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
    matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)
    # date_item_city_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_item_city_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
    matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
    matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)
    #date_type_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_type_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
    matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
    matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)
    # date_subtype_avg_item_cnt_lag_1
    group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_subtype_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
    matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
    matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)
    print(2)
    #Trend features
    group = train.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['item_id'], how='left')
    matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

    group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
    matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

    lags = [1,2,3,4,5,6]
    matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

    for i in lags:
        matrix['delta_price_lag_'+str(i)] = \
            (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

    def select_trend(row):
        for i in lags:
            if row['delta_price_lag_'+str(i)]:
                return row['delta_price_lag_'+str(i)]
        return 0
    
    matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
    matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_price_lag'].fillna(0, inplace=True)

    fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for i in lags:
        fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
        fetures_to_drop += ['delta_price_lag_'+str(i)]

    matrix.drop(fetures_to_drop, axis=1, inplace=True)

    #Last month shop revenue trend
    group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
    group.columns = ['date_shop_revenue']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
    matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

    group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
    group.columns = ['shop_avg_revenue']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
    matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

    matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
    matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

    matrix = lag_feature(matrix, [1], 'delta_revenue')

    matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)

    #Special features
    matrix['month'] = matrix['date_block_num'] % 12
    days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
    matrix['days'] = matrix['month'].map(days).astype(np.int8)
    #last_sale
    #item_shop_last_sale
    cache = {}
    matrix['item_shop_last_sale'] = -1
    matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():    
        key = str(row.item_id)+' '+str(row.shop_id)
        if key not in cache:
            if row.item_cnt_month!=0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num         
    #item_last_sale
    cache = {}
    matrix['item_last_sale'] = -1
    matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():    
        key = row.item_id
        if key not in cache:
            if row.item_cnt_month!=0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            if row.date_block_num>last_date_block_num:
                matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
                cache[key] = row.date_block_num    
    #first_sale
    matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
    matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

    #Final
    matrix = matrix[matrix.date_block_num > 11]

    #NaN fill zero
    def fill_na(df):
        for col in df.columns:
            if ('_lag_' in col) & (df[col].isnull().any()):
                if ('item_cnt' in col):
                    df[col].fillna(0, inplace=True)         
        return df

    matrix = fill_na(matrix)

    print(time.time() - ts)

    print(matrix.columns)
    print(matrix.info())
    print(matrix.shape)

    matrix.to_pickle('data.pkl')
    del matrix
    del cache
    del group
    del items
    del shops
    del cats
    del train

    print('start train')
    #gc.collect()
def train1(k):

    test  = pd.read_csv('./Data/test.csv').set_index('ID')
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 10, 'shop_id'] = 11
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)


    ts = time.time()
    data = pd.read_pickle('data.pkl')
    data = data[[
        'date_block_num',
        'shop_id',
        'item_id',
        'item_cnt_month',
        #'ID',
        'city_code',
        'item_category_id',
        'type_code',
        'subtype_code',
        'item_cnt_month_lag_1',
        'item_cnt_month_lag_2',
        'item_cnt_month_lag_3',
        'item_cnt_month_lag_6',
        'item_cnt_month_lag_12',
        'date_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_2',
        'date_item_avg_item_cnt_lag_3',
        'date_item_avg_item_cnt_lag_6',
        'date_item_avg_item_cnt_lag_12',
        'date_shop_avg_item_cnt_lag_1',
        'date_shop_avg_item_cnt_lag_2',
        'date_shop_avg_item_cnt_lag_3',
        'date_shop_avg_item_cnt_lag_6',
        'date_shop_avg_item_cnt_lag_12',
        'date_cat_avg_item_cnt_lag_1',
        'date_shop_cat_avg_item_cnt_lag_1',
        'date_shop_type_avg_item_cnt_lag_1',
        'date_shop_subtype_avg_item_cnt_lag_1',
        'date_city_avg_item_cnt_lag_1',
        'date_item_city_avg_item_cnt_lag_1',
        'date_type_avg_item_cnt_lag_1',
        'date_subtype_avg_item_cnt_lag_1',
        'delta_price_lag',
        'month',
        'days',
        'item_shop_last_sale',
        'item_last_sale',
        'item_shop_first_sale',
        'item_first_sale',
    ]]

    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    del data
    
    model = XGBRegressor(
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300, 
        colsample_bytree=0.8, 
        subsample=0.8, 
        eta=0.3,    
        seed=42)

    model.fit(
        X_train, 
        Y_train, 
        eval_metric="rmse", 
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
        verbose=True, 
        early_stopping_rounds = 10)
    print('it OK')
    Y_pred = model.predict(X_valid).clip(0, 20)
    Y_test = model.predict(X_test).clip(0, 20)

    submission = pd.DataFrame({
        "ID": test.index, 
        "item_cnt_month": Y_test
    })
    submission.to_csv('xgb_submission'+ str(k)+'.csv', index=False)

    # save predictions for an ensemble
    pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
    pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))
    print(time.time() - ts)

preprocessing()
number = 0
train1(number)
