import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import numpy as np


#For data:
def dataExtracterMonths(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate)
    data = data.reset_index()[["Date", "Open", "High", "Low", "Close"]]
    data.columns = data.columns.droplevel(1)
    data.columns.name = None
    # Convert 'Date' column to datetime type
    data['Date'] = pd.to_datetime(data['Date'])

    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)

    # Resample the data to monthly frequency
    obs = data.resample('ME').agg({'Open': 'first','High': 'max','Low': 'min','Close': 'last'})

    # Reset the index to have 'Date' as a column again
    obs = obs.reset_index()

    # --- Convert dates to just YYYY-MM-DD ---
    obs['Date'] = obs['Date'].dt.date  # <-- this removes the timestamp
    print(f"The dataset has observations across {len(obs)} months")
    return obs

def dataExtracterDays(ticker, startDate, endDate):
    data = yf.download(ticker, start=startDate, end=endDate)
    data = data.reset_index()[["Date", "Open", "High", "Low", "Close"]]
    data.columns = data.columns.droplevel(1)
    data.columns.name = None
    # Convert 'Date' column to datetime type
    obs = data

    # --- Convert dates to just YYYY-MM-DD ---
    obs['Date'] = obs['Date'].dt.date  # <-- this removes the timestamp
    print(f"The dataset has observations across {len(obs)} days")
    return obs

def dataCutter(obs, train_share):
    split_index = int(len(obs) * train_share)
    train = obs[:split_index]

    start_date = obs['Date'][split_index]
    print("The training data ends on: ", start_date)
    
    obs = obs[obs.columns[1:5]]
    train = train[train.columns[1:5]]

    predict_size = len(obs)-len(train)
    print("We are trying to predict the next ", predict_size, " units of time (e.g. days, months...).")

    return obs, train, start_date, predict_size


#To predict:
def HMMPricePredictor(data, obs, window_size, Ncomp, use_log=False):
    #If use_log, predict log of close price, otherwise predict close price directly
    if use_log:
        data = data.copy()
        obs = obs.copy()
        data['Close'] = np.log(data['Close'])
        obs['Close'] = np.log(obs['Close'])

    # Calculate number of rows and set training window
    T = data.shape[0]

    # Define the size of the training window
    predict_size = len(obs) - len(data) # Data points to predict
    hmm_price = []

    temp_T = T
    first_time = True

    # Sliding window approach to predict future prices
    while T < temp_T + predict_size:

        # Train HMM on data from T-window_size+1 to T
        train_data = obs.iloc[T-window_size:T]
        train_data = train_data.dropna()

        # Set the random seed
        np.random.seed(1)

        if(first_time):
            first_time = False
            model = hmm.GaussianHMM(n_components=Ncomp)
        else:
            old_model= model
            model = hmm.GaussianHMM(n_components=Ncomp, init_params="c")
            model.startprob_ = old_model.startprob_
            model.transmat_ = old_model.transmat_
            model.means_ = old_model.means_

        model.fit(train_data)

        # Calculate original likelihood
        original_likelihood = model.score(train_data)

        # Loop to find new likelihood
        t=T
        min_diff = float('inf')
        min_t = T
        min_likelihood = original_likelihood
        while t-window_size>  0:
            t = t-1

            train_data = obs.iloc[t-window_size:t]
            new_likelihood = model.score(train_data)
            if (abs(new_likelihood - original_likelihood))< min_diff:  # Threshold for comparison by choosing that new_likelihood which is minimum
                min_diff = abs(new_likelihood - original_likelihood)
                min_t = t
                min_likelihood = new_likelihood

        # Calculate the predicted close price
        close_price = obs['Close'][T-1] + ((obs['Close'][min_t + 1] - obs['Close'][min_t]) * np.sign(original_likelihood - min_likelihood))

        hmm_price.append(close_price)
        T=T+1

    close = []
    truncated_obs = obs.iloc[T-predict_size:T]
    for i in truncated_obs['Close']:
        close.append(i)

    if use_log:
        hmm_price = np.exp(hmm_price)
        close = np.exp(close)
        
    return hmm_price, close


#From Sagar, to evaluate:
# 1. Absolute Percentage Error (APE)
def ape(real_, pred_):
    APE = 0
    sum = 0
    N = len(real_)
    # Calculate the sum of absolute differences between real and predicted values
    for i in range(1, N):
        sum += (np.abs(real_[i] - pred_[i])) / N

    # Calculate APE as a ratio of the sum to the mean of real values
    APE = sum / (np.mean(real_))

    return APE

# 2. Average Absolute Error (AAE)
def aae(real_, pred_):
  AAE = 0
  sum = 0
  N = len(real_)
  for i in range(1,N):
    sum += (np.abs(real_[i] - pred_[i]))/N
  AAE = sum
  return AAE


# 3. Average Relative Percentage Error (ARPE)
def arpe(real_, pred_):

  sum = 0
  N = len(real_)
  for i in range(1,N):
    sum += (np.abs(real_[i] - pred_[i]))/N
  ARPE = sum/N
  return ARPE

# 4. Root Mean Squared Error (RMSE)
def rmse(real_, pred_):
  sum = 0
  N = len(real_)
  for i in range(1,N):
    sum += (np.square(real_[i] - pred_[i]))/N
  RMSE = np.sqrt(sum)
  return RMSE


