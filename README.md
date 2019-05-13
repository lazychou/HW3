##kaggle的Predict Future Sales
#目標: 
predict total sales for every product and store in the next month

##資料
items.csv
shops.csv
item_categories.csv
sales_train.csv
test.csv

##資料前處理
#heal data and remove outliers
sales_train.csv中的item_cnt_day和item_price有outliers，將其去除。
item_price只取小於100000
item_cnt_day只取小於1001

item_price有數據是負數，這是不合理的，所以用該商店該商品該月的平均價格取代。

有部分商店重複，所以將重複的商店統一命名。

#work with shops/items/cats objects and features




#create matrix as product of item/shop pairs within each month in the train set
get monthly sales for each item/shop pair in the train set and merge it to the matrix
clip item_cnt_month by (0,20)
append test to the matrix, fill 34 month nans with zeros
merge shops/items/cats to the matrix
add target lag features
add mean encoded features
add price trend features
add month
add days
add months since last sale/months since first sale features
cut first year and drop columns which can not be calculated for the test set
select best features
set validation strategy 34 test, 33 validation, less than 33 train
fit the model, predict and clip targets for the test set
