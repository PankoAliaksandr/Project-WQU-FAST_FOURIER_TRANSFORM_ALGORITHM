#Mini Project 2 FAST FOURIER TRANSFORM ALGORITHM

# Libraries
import numpy as np
import scipy.signal as sc
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime

# Download stock prices 
start_date=datetime.date(2013, 7,10)
end_date=datetime.date(2017, 8, 1)
df = web.DataReader('AGU', 'google', start_date, end_date)
close_prices = df.Close
days_number = np.count_nonzero(close_prices)
print("number of days = ", days_number)
plt.figure(1)
plt.plot(close_prices)
plt.title('Agrium stock price movement')

# Detrend 
detrended_prices=sc.detrend(close_prices)
plt.figure(2)
plt.plot(detrended_prices)
plt.title('Agrium stock detrended prices') 

# Smooth detrended Apple stock prices
w=np.blackman(20) #we selected 20 the parameter of the blackman window function
prices_smooth=np.convolve(w/w.sum(),detrended_prices,mode='same')
plt.figure(3)
plt.plot(prices_smooth)
plt.title('Blackman window function for detrended Agrium stock price') 

# Apply FFT Algorithm 
fft = np.abs(np.fft.rfft(prices_smooth))
print(fft)
plt.figure(4)
plt.plot(fft)
plt.title('FFT Algorithm applied to Agrium stock price')

# Adjust the scale
plt.figure(5)
plt.plot(fft)
plt.title('FFT Algorithm applied to Agrium stock price. Scaled')
plt.ylim([0,3000])
plt.xlim([0,10])
