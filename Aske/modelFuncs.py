import pandas as pd
import yfinance as yf

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
    data['Date'] = pd.to_datetime(data['Date'])

    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)

    # Resample the data to monthly frequency
    obs = data.resample('D').agg({'Open': 'first','High': 'max','Low': 'min','Close': 'last'})

    # Reset the index to have 'Date' as a column again
    obs = obs.reset_index()

    # --- Convert dates to just YYYY-MM-DD ---
    obs['Date'] = obs['Date'].dt.date  # <-- this removes the timestamp
    print(f"The dataset has observations across {len(obs)} days")
    return obs