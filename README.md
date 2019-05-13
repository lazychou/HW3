## kaggle的Predict Future Sales
# 目標:
predict total sales for every product and store in the next month

## 資料
items.csv<br>
shops.csv<br>
item_categories.csv<br>
sales_train.csv<br>
test.csv<br>

## 資料前處理
# heal data and remove outliers
sales_train.csv中的item_cnt_day和item_price有outliers，將其去除。<br>
item_price只取小於100000<br>
item_cnt_day只取小於1001<br>

item_price有數據是負數，這是不合理的，所以用該商店該商品該月的平均價格取代。

有部分商店重複，所以將重複的商店統一命名。

# work with shops/items/cats objects and features,get monthly sales for each item/shop pair
因為shops.csv的shop_name包含了城市與店名，因此對其進行分割，產生兩個新的屬性shop_id和city_code<br>
item_categories.csv的item_category_name分割成item_category_id、type_code和subtype_code三個屬性<br>
因為sales_train.csv的數據是以天為單位，所以按照物品、商店將其統整成該商店該物品每月銷售量，然後clip item_cnt_month by (0,20)<br>

# append test to the matrix, fill 34 month nans with zeros
將所要預測的月的date_block_num訂為34

# add target lag features
建立函數lag_feature用於建立屬性表示某個屬性前n個月的數值

# add mean encoded features
將item_cnt依照商、商品、商店和商品等禁行平均所得數值。
# add price trend features
將上前六個月的月平均銷售作為特徵
# add month、days
整理時間相關特徵
# add months since last sale/months since first sale features
加入最後價格和最初價格為特徵

## 訓練
使用xgboost的XGBRegressor<br>
總共有39個特徵<br>
有些實驗是只挑選部分特徵來訓練<br>

# 模型參數
max_depth=8,<br>
n_estimators=1000,<br>
min_child_weight=300, <br>
colsample_bytree=0.8, <br>
subsample=0.8, <br>
eta=0.3,    <br>
seed=42<br>

## 結果
1.23646(以範例所得到)<br>
0.91300<br>
0.90646<br>
0.90684
