import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import scipy as sc
yf.pdr_override()

def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks,  start=start, end=end)
    stockData = stockData['Close']
    stockData = stockData[1:]

    returns = stockData.pct_change()
    mean = returns.mean()
    covMat = returns.cov()

    return mean, covMat

def portPerf(weights, meanReturns, covMatrix):
    returns = np.mean(meanReturns * weights) * 252
    std = np.sqrt(np.dot(np.transpose(weights), np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std

def negSharpe(weights, meanReturns, covMatrix, riskFree = 0):
    rets, std = portPerf(weights, meanReturns, covMatrix)
    return - ((rets - riskFree) / std)

def maxSR(meanRets, covMatrix, constraintSet = (0, 1), riskFree = 0):
    numAssets = len(meanRets)
    args = (meanRets, covMatrix, riskFree)
    constraints = ({'type': "eq", 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(negSharpe, numAssets*[1./numAssets], args=args, method='SLSQP', 
                         bounds = bounds, constraints=constraints)
    return result

def portfolioVar(weights, mean, cov):
    return portPerf(weights, mean, cov)[1]

def minimizeVar(meanRets, covMatrix, constraintSet = (0, 1), riskFree = 0):
    numAssets = len(meanRets)
    args = (meanRets, covMatrix)
    constraints = ({'type': "eq", 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(portfolioVar, numAssets*[1./numAssets], args=args, method='SLSQP', 
                         bounds = bounds, constraints=constraints)
    return result



stockList = ['AMZN', 'V', 'GS']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

data = getData(stockList, start=startDate, end=endDate)
print(data)

weights = [0.5, 0.3, 0.2]

performace = portPerf(weights, data[0], data[1])

print(f'{round(performace[0] * 100, 2)}%')
print(f'std: {round(performace[1], 2)}')
print()

finalWeights = minimizeVar(data[0], data[1])
maxSR = finalWeights['fun']
resultWeights = finalWeights['x']
otherSharpe = -(negSharpe(resultWeights, data[0], data[1]))

print(finalWeights)
print(maxSR)
print(otherSharpe)
