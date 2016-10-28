import pandas as pd
import pickle

# number of records to be loaded
NUM_RECORDS = 100000

# first 0.8 M records are negative, the rest 0.8 M records are positive
# number of records to be skipped
NUM_SKIP = 800000 - NUM_RECORDS//2

# load the data
data = pd.read_csv('data/training.1600000.processed.noemoticon.csv'
                   , nrows=NUM_RECORDS, usecols=[0, 5], header=None, names=['score', 'tweet']
                   ,skiprows=NUM_SKIP, encoding='iso-8859-2')

# save the data in pickle to allow test work
with open('data/sample_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print('#'*50)
print("shape of the dataset is {}".format(data.shape))
print()
print("first few records")
print(data.head())

print(data.score.value_counts())




