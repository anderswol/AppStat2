import numpy as np

# 1. Mean Absolute Percentage Error (MAPE)
def mape(real_, pred_):
  real_ = np.asarray(real_).flatten()
  pred_ = np.asarray(pred_).flatten()
  
  assert real_.shape == pred_.shape  # Ensure real_ and pred_ have the same shape
  mask = real_ != 0
  real_, pred_ = real_[mask], pred_[mask]
  
  return np.mean(np.abs((real_-pred_)/real_))

# 2. Root Mean Squared Error (RMSE)
def rmse(real_, pred_):
    real_ = np.asarray(real_).flatten()
    pred_ = np.asarray(pred_).flatten()
    
    assert real_.shape == pred_.shape  # Ensure real_ and pred_ have the same shape
    
    return np.sqrt(np.mean((real_ - pred_)**2))
  
# 3. Directional Accuracy
def direction_accuracy(real_, pred_):
  real_ = np.asarray(real_).flatten()
  pred_ = np.asarray(pred_).flatten()
  
  assert real_.shape == pred_.shape  # Ensure real_ and pred_ have the same shape
  
  directReal = np.sign(np.diff(real_))
  directPred = np.sign(np.diff(pred_))
  direction_accuracy = np.mean(directReal == directPred)
  return direction_accuracy

# 4. Average Absolute Error (AAE)
def aae(real_, pred_):
  real_ = np.asarray(real_).flatten()
  pred_ = np.asarray(pred_).flatten()
  AAE = 0
  sum = 0
  N = len(real_)
  for i in range(1,N):
    sum += (np.abs(real_[i] - pred_[i]))/N
  AAE = sum
  return AAE


# 5. Average Relative Percentage Error (ARPE)
def arpe(real_, pred_):
  real_ = np.asarray(real_).flatten()
  pred_ = np.asarray(pred_).flatten()
  sum = 0
  N = len(real_)
  for i in range(1,N):
    sum += (np.abs(real_[i] - pred_[i]))/N
  ARPE = sum/N
  return ARPE


