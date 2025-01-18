import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt



# let's start with loading the data
file_path = "stock_data.csv"  
df = pd.read_csv(file_path, skiprows=2)  # we have to skip the first two rows beacuse they are not needed

# Now we have to rename the columns 
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Date are in string format we have to change it to dateTime
df['Date'] = pd.to_datetime(df['Date'])

df = df.dropna()

#now lets see our data
# print(df.head())
# print(df.columns)
# print(df.describe())


#lets look for the outliers

# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(10, 6))
# sns.histplot(df['Volume'], kde=True, color='blue')
# plt.title('Distribution of Close Prices')
# plt.xlabel('Close Price')
# plt.ylabel('Frequency')
# plt.show()


# sns.boxplot(x=df['Open'], color='blue')
# plt.title('Boxplot of Close Prices')
# plt.xlabel('Close Price')
# plt.show()
  
 #lets remove the outliers
q1  = df['Volume'].quantile(0.25)
q2 = df['Volume'].quantile(0.75)

iqr = q2 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q2 + 1.5 * iqr

df = df[(df['Volume'] > lower_bound) & (df['Volume'] < upper_bound)]



# sns.boxplot(x=df['Open'], color='blue')
# plt.title('Boxplot of Close Prices')
# plt.xlabel('Volume')
# plt.show()

#we are done wiht outliers

#Now lets deal with multicollinearity
#to deal with multicollinearity we do some feature engineering

df['Average_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['Price_Range'] = df['High'] - df['Low']
df = df.drop(['High', 'Low', 'Open' , 'Date'], axis=1)
print(df.describe())


#lets check the correlation

# plt.figure(figsize=(10, 6))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()



target = df['Close']
features = ['Volume',  'Price_Range'] # we ahve to remove date because it is not needed and 

# setting up the model
X = df[features]
y = target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# let's make predictions now
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#lets fit our model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

#now we are done lets print rmse value
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print('Root Mean Squared Error:', rmse)