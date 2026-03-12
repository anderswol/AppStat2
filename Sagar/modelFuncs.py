import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import numpy as np

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


def HMMPricePredictor(data, obs, window_size, Ncomp):
    # Calculate number of rows and set training window
    T = data.shape[0]
    # print("T= ", T)

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
        np.random.seed(123)

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

    # Print the calculated prices
    # print("HMM Prices: ")
    # print(hmm_price)

    close = []
    truncated_obs = obs.iloc[T-predict_size:T]
    for i in truncated_obs['Close']:
        close.append(i)
    return hmm_price, close
