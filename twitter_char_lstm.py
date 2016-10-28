import pickle
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

from twtitter_char_dict import Corpus

corpus = Corpus('data/sample_data.pkl')

with open('data/sample_data.pkl', 'rb') as f:
    data = pickle.load(f)

data['score'] = data['score'].apply(lambda x: 1 if x > 0 else 0)

np.random.seed(12)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data.iloc[indices, :]
data.reset_index(drop=True, inplace=True)

nb_train = int(0.7 * data.shape[0])

train_x = data.loc[:nb_train, 'tweet'].values
train_y = data.loc[:nb_train, 'score'].values

test_x = data.loc[nb_train:, 'tweet'].values
test_y = data.loc[nb_train:, 'score'].values


EMBEDDING_LENGTH = len(corpus.charset) + 1
TWEET_LENGTH = 150
EMBEDDING_OUTPUT_DIM = 30

X_Train = corpus.code_sentense(train_x)
X_Train = pad_sequences(X_Train, maxlen=TWEET_LENGTH, padding='post')

Y_Train = train_y


model = Sequential()
model.add(Embedding(
    input_dim=EMBEDDING_LENGTH,
    output_dim=EMBEDDING_OUTPUT_DIM,
    input_length=TWEET_LENGTH,
    mask_zero=True
))
model.add(LSTM(
    output_dim=50,
    return_sequences=False
))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_Train, Y_Train, batch_size=32, nb_epoch=15, verbose=2)

