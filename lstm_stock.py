#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import fbprophet
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

#creating dataframe
# rms = sqrt(mean_squared_error(test.Count, y_hat_avg.avg_forecast))
# print(rms)
start = datetime.datetime(2018, 1, 1)
end = datetime.date.today()

data = pdr.get_data_yahoo('GOOG',start,end)
print(data.head(2))

dch=[]
print('row : '+str(len(data))+ ', Close : '+ str(data['Close'][0]) )
for row in range(len(data)):
    #daily_change['Daily Change'].append(data['Close'][row]-data['Open'][row])
    dch.append(data['Close'][row]-data['Open'][row])

data['Daily Change']=dch
#data['Scaled DChange']=dch.pct_change()
#data['Scaled Volume']=data['Volume'].pct_change()
cp_df=pd.DataFrame(data, columns = ['Daily Change','Volume'])
cp_rets = cp_df.pct_change()

# create the average data colume charts ++++++++++++++++++++
ma_day = [10,20,50,200]

for ma in ma_day:
	column_name = "MA for %s days" %(str(ma))
#    AAPL[column_name] = pd.rolling_mean(AAPL['Adj Close'], ma)
	data[column_name]= data['Adj Close'].rolling(ma).mean()



#GOOG['Volume'].plot(legend=True,figsize=(10,4))
#data[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days', 'MA for 200 days']].plot(subplots=False,figsize=(10,4))

#-----------------------------------------------------------
# dchange vs volumn +++++++++++++++++++++++++
#sns.jointplot('Daily Change','Volume',cp_rets,kind='scatter',color='seagreen')
#sns.pairplot(cp_rets.dropna())
# dchange+++++++++++++
#fig, ax = plt.subplots()
#fig.set_size_inches(1.7, 8.27)
#ax = sns.boxplot(data=data['Daily Change'])
#plt.xticks('Daily Change')

#data['Daily Change'].plot(legend=True,figsize=(10,4))
#----------------------------
x = data.index.date
print(data.describe())
# plot the close price
# data['Adj Close'].plot()
# compare bar chart with trend line +++++++++++++++++++++++++
#x = data.index.date
#data.plot( y=['Volume','Daily Change'],  kind='bar')
# ------------------------------


adc_mean=data['Adj Close'].mean()
adc_se=data['Adj Close'].sem()*1.96
doublediff = np.diff(np.sign(np.diff(data['Adj Close'])))
peak_locations = np.where(doublediff == -2)[0] + 1
doublediff2 = np.diff(np.sign(np.diff(-1*data['Adj Close'])))
trough_locations = np.where(doublediff2 == -2)[0] + 1

print('mean: '+ str(adc_mean)+ ', se: '+ str(adc_se))

# ++++++++++++++++++++++charts block needed
#plt.figure(figsize=(16,10), dpi= 80)
#
#
##plt.plot(x, data['Adj Close'], color="Black", lw=2, label='Price')
#plt.scatter(x[peak_locations], data['Adj Close'][peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
#plt.scatter(x[trough_locations], data['Adj Close'][trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')
#data[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days', 'MA for 200 days']].plot(legend=True,subplots=False,figsize=(10,4))
### Annotate
##for t, p in zip(trough_locations[1::5], peak_locations[::3]):
##    plt.text(x[p], data['Adj Close'][p]+15, x[p], horizontalalignment='center', color='darkgreen')
##    plt.text(x[t], data['Adj Close'][t]-35, x[t], horizontalalignment='center', color='darkred')
##
### Decoration
##plt.ylim(50,750)
##xtick_location = x.tolist()[::6]
##xtick_labels = x.tolist()[::6]
##plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
##plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
##plt.yticks(fontsize=12, alpha=.7)
#
#
#plt.fill_between(x, data['Adj Close'] - adc_se, data['Adj Close'] + adc_se, color='Cyan', label='Standard Error')
#plt.legend()
# -----------------------------
#data.info()
#print(data.head())

#for i in range(0,len(data)):
#    new_data['Date'][i] = data['Date'][i]
#    new_data['Close'][i] = data['Close'][i]

#setting index
#new_data.index = new_data.Date
#new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
ci=0.95
forecast_period = int(len(data)*0.8)
print("total=== "+  str(len(data)) +  "  train==="   + str(forecast_period))

#remove --------------
#new_data=pd.DataFrame(index=range(0,len(data)), columns=['Date','Adj Close'])
#for i in range(0,len(data)):
#    new_data['Date'][i] = data.index[i]
#    new_data['Adj Close'][i] = data['Adj Close'][i]

#new_data1=data
#setting index
#new_data.index = new_data.Date
#new_data.drop('Date', axis=1, inplace=True)
# ---------------------
#creating train and test sets
#dataset = new_data.values
dataset = data.iloc[:, 4:5].values
#dataset = data['Adj Close'].reset_index().drop(['Date'],axis=1)
#dataset = data['Adj Close']




train = data[0:forecast_period]
valid = data[forecast_period:]
y_train=train['Adj Close']
x_train=train.drop(['Adj Close'], axis=1)
y_valid=valid['Adj Close']
x_valid=valid.drop(['Adj Close'], axis=1)

print(x_valid.head(2))
print(y_valid.head(2))
# converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
data['Scaled Aclose']=scaled_data

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
print(len(scaled_data))

print(x_train)
print('=====================')
print(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
print(x_train)
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
new_data=data.iloc[:,4:5]
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
#print(closing_price)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print('rms = '+ str(rms))

train = new_data[:forecast_period]
valid = new_data[forecast_period:]
valid['Predictions'] = closing_price

# Calculate the differences between consecutive measurements
valid['pred_diff']=valid['Predictions'].diff()
valid['real_diff']=valid['Adj Close'].diff()
# Correct is when we predicted the correct direction
valid['correct'] = (np.sign(valid['pred_diff'][1:])==np.sign(valid['real_diff'][1:]))*1
# Accuracy when we predict increase and decrease
increase_accuracy = 100 * np.mean(valid[valid['pred_diff'] > 0]['correct'])
decrease_accuracy = 100 * np.mean(valid[valid['pred_diff'] < 0]['correct'])
# Calculate percentage of time actual value within prediction range
valid['in_range'] = False

for i in valid.index:
    if (valid.loc[i, 'Predictions'] < data.loc[i, 'High']) & (valid.loc[i, 'Predictions'] > data.loc[i, 'Low']):
        valid.loc[i, 'in_range'] = True

in_range_accuracy = 100 * np.mean(valid['in_range'])

print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))

data['Predictions']= valid['Predictions']
#print('The actual value was within the {:d}% confidence interval {:.2f}% of the time.'.format(int(100 * model.interval_width), in_range_accuracy))
print(valid.describe())

#plt.plot(train['Adj Close'])
#plt.plot(valid['Predictions'], label='Predictions')
#plt.plot(data[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days', 'MA for 200 days']], ['Price', 'MA for 10 days','MA for 20 days','MA for 50 days', 'MA for 200 days'])
data[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days', 'MA for 200 days', 'Predictions']].plot(legend=True,subplots=False,figsize=(10,4),grid=True)
plt.fill_between(x, data['Predictions'] - adc_se, data['Predictions'] + adc_se, color='Cyan', label='Standard Error')
#plt.plot(legend=True,subplots=False,figsize=(10,4))

# predict for future price  ++++++++++++
#train = dataset.tail(forecast_period)
#model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
#future = m.make_future_dataframe(periods=30, freq="H")
#forecast = m.predict(future)
#fig1 = m.plot(forecast)
pairs=('adc_se', '')
attr=[adc_se, increase_accuracy]
# ---------------------
data.to_csv(r'stock1.csv')
attr.to_csv(r'prediction.csv')
plt.show()


