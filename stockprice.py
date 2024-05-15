import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df=pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/BitcoinPricePrediction.csv')
df.isna().sum()
df.head()
print(df.columns)

y=df['next_day_closing_price']
x=df[['opening_price', 'highest_price', 'lowest_price',
       'closing_price', 'transactions_in_blockchain', 'avg_block_size',
       'sent_coins_in_usd', 'avg_transaction_fees',
       'median_transaction_fees', 'avg_block_time', 'avg_transaction_value',
       'median_transaction_value', 'tweets', 'google_trends','number_of_coins_in_circulation']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=2529)

model=KNeighborsRegressor(n_neighbors=6)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(y_pred)
print("error",mean_absolute_percentage_error(y_test,y_pred))
