import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


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

def HMM_predict(obsData, train_size, window_size, Ncomp, doprint=False):
    rng = np.random.default_rng(123)
    obs = obsData.copy()
    train = obs[:train_size]
    features = ['LogReturn', 'Volatility', 'Momentum']
    train_data = train[features].values
    
    #Normalize data:
    train_scaled = scaler.fit_transform(train_data)
    # Train global HMM
    model = hmm.GaussianHMM(n_components=Ncomp, random_state=123)
    model.fit(train_scaled)
    
    means = model.means_
    stds = np.sqrt(model._covars_)
    first = True
    if doprint:
        print("State-wise means:", np.round(means.flatten(), 3))
        print("State-wise std:", stds.flatten())
        print("State counts:", np.bincount(model.predict(train_scaled)))
        print(np.sum(np.isnan(train_data)))
        print(np.round(model.transmat_, 2))
        first = True
    
    T = max(window_size, train_size)
    predictions = []
    returns = []
    expected_pct_changes = []
    while T < len(obs):
        window = obs[features].iloc[T-window_size:T].values
        window_scaled = scaler.transform(window)

        # Predict the current state based on the most recent window
        current_state = model.predict(window_scaled)[-1]

        # Choose the most likely next state based on transition probabilities
        next_state = rng.choice(np.where(model.transmat_[current_state] == np.max(model.transmat_[current_state]))[0])  

        # Sample a return from the predicted next state
        predicted_return_scaled = rng.normal(loc=means[next_state][0], scale=stds[next_state][0])

        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = predicted_return_scaled

        predicted_log_return = scaler.inverse_transform(dummy)[0, 0]
        

        # Convert to price
        last_price = obs['Close'].iloc[T-1]
        predicted_price = last_price * np.exp(predicted_log_return)
        if first:
            print(f"Date for day before first prediction: {obs['Date'].iloc[T-1]}")
            print(f"Date for first day of real data: {obs['Date'].iloc[train_size]}")
            first = False

        returns.append(predicted_log_return)
        predictions.append(predicted_price)
        expected_pct_changes.append(np.exp(predicted_log_return) - 1)
        T += 1

    close = obs['Close'].iloc[train_size:].values

    return predictions, close, returns, expected_pct_changes