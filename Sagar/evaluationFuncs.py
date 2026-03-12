import numpy as np

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

def direction_accuracy(real_, pred_):
  directReal = np.sign(np.diff(real_))
  directPred = np.sign(np.diff(pred_))
  direction_accuracy = np.mean(directReal == directPred)
  return direction_accuracy
