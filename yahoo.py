
#Let's go ahead and start with some imports
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import datetime
# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from IPython.display import SVG

# For reading stock data from yahoo
# from pandas.io.data import DataReader
# from pandas_datareader import DataReader
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
# For time stamps
#from datetime import datetime

# For division
#from __future__ import division




# The tech stocks we'll use for this analysis
tech_list = ['AAPL','GOOG','MSFT','AMZN']

# Set up End and Start times for data grab
#end = datetime.now()
#start = datetime(end.year - 1,end.month,end.day)

start = datetime.datetime(2017,12,31)
end = datetime.date.today()

#For loop for grabing yahoo finance data and setting as a dataframe

for stock in tech_list:
    # Set DataFrame as the Stock Ticker
    #globals()[stock] = pdr.DataReader(stock,'yahoo',start,end)
    # globals()[stock] = pdr.get_data_yahoo(stock,start,end)
    print(stock + '------')
    globals()[stock] = pdr.get_data_yahoo(stock,start,end)
    print(globals()[stock].describe())
    globals()[stock].info()
#    globals()[stock]['Volume'].plot(legend=True,figsize=(10,4))
#    globals()[stock]['Adj Close'].plot(legend=True,figsize=(10,4))


# describe bRelations  +++++++++++++++

#ma_day = [10,20,50,200]
#
#for ma in ma_day:
#	column_name = "MA for %s days" %(str(ma))
##    AAPL[column_name] = pd.rolling_mean(AAPL['Adj Close'], ma)
#	GOOG[column_name]= GOOG['Adj Close'].rolling(ma).mean()
#
#
#
#GOOG['Volume'].plot(legend=True,figsize=(10,4))
#GOOG[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days', 'MA for 200 days']].plot(subplots=False,figsize=(10,4))

#-----------------------------------------------

#AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
#AAPL['Daily Return'].plot(figsize=(12,4),legend=True,linestyle='--',marker='o')
#AAPL['Daily Return'].hist()
# dataset comparisons:
closing_df=pdr.get_data_yahoo(["AAPL","GOOG","MSFT","AMZN"],start,end)['Adj Close']
tech_rets = closing_df.pct_change()
#sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter',color='seagreen')
#sns.pairplot(tech_rets.dropna())

#returns_fig = sns.PairGrid(tech_rets.dropna())


# kde & scater +++++++++++++++++
#returns_fig = sns.PairGrid(closing_df)
#returns_fig.map_upper(plt.scatter,color='purple')
#returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
#returns_fig.map_diag(plt.hist,bins=30)
#--------------------------------

#sns.heatmap(tech_rets.corr(),annot=True)


# risk analysis +++++++++++++++++++++++++
#rets = tech_rets.dropna()
#
#area = np.pi*20
#
#plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)
#
## Set the x and y limits of the plot (optional, remove this if you don't see anything in your plot)
#plt.ylim([0.01,0.025])
#plt.xlim([-0.003,0.004])
#
##Set the plot axis titles
#plt.xlabel('Expected returns')
#plt.ylabel('Risk')
#
## Label the scatter plots, for more info on how this is done, chekc out the link below
## http://matplotlib.org/users/annotations_guide.html
#for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
#   plt.annotate(
#        label,
#        xy = (x, y), xytext = (50, 50),
#        textcoords = 'offset points', ha = 'right', va = 'bottom',
#        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
#--------------------------------------
# confident of daily loss Value at risk using the "bootstrap" method
# sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
# print(rets['AAPL'].quantile(0.05))

####Value at Risk using the Monte Carlo method

# Set up our time horizon
days = 365

# Now our delta
dt = 1/days

# Now let's grab our mu (drift) from the expected return data we got for AAPL
mu = rets.mean()['GOOG']

# Now let's grab the volatility of the stock from the std() of the average return
sigma = rets.std()['GOOG']


    


def stock_monte_carlo(start_price, days, mu, sigma):
#    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''

    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)

    # Run price array for number of days
    for x in range(1, days):
        # Calculate Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x - 1] + (price[x - 1] * (drift[x] + shock[x]))

    return price

#start_price = 569.85
#
##for run in range(10):
##    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
#
##plt.xlabel("Days")
##plt.ylabel("Price")
##plt.title('Monte Carlo Analysis for Google')
#
## Set a large numebr of runs
#runs = 10000
#
## Create an empty matrix to hold the end price data
#simulations = np.zeros(runs)
#
## Set the print options of numpy to only display 0-5 points from an array to suppress output
#np.set_printoptions(threshold=5)
#
#for run in range(runs):
#    # Set the simulation data point as the last stock price for that run
#    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];
#
## Now we'lll define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
#q = np.percentile(simulations, 1)
#
## Now let's plot the distribution of the end prices
#plt.hist(simulations, bins=200)
#
## Using plt.figtext to fill in some additional information onto the plot
#
## Starting Price
#plt.figtext(0.6, 0.8, s="Start price: $%.2f" % start_price)
## Mean ending price
#plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())
#
## Variance of the price (within 99% confidence interval)
#plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))
#
## Display 1% quantile
#plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)
#
## Plot a line at the 1% quantile result
#plt.axvline(x=q, linewidth=4, color='r')
#
## Title
#plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');
#
#plt.show()
    


# google describe() box chart++++++++++++++++
#dataG=GOOG
#print(dataG.head(2))
##dataG=dataG.drop(['Volume'], axis=1)
#print(dataG.describe())
#fig, ax = plt.subplots()
#fig.set_size_inches(1.7, 8.27)
#ax = sns.boxplot(data=GOOG['Volume'])
#plt.xticks('Volume')
##hi_pivots = GOOG.pivot_table()
##hi_pivots = GOOG.pivot_table(index='flight_date', columns='origin', values='arr_delay')
##GOOG.plot(kind='box', figsize=[16,8])
# ------------------------------------------
plt.show()