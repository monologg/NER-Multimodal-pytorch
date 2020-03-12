from tensorflow.keras.layers import LSTM, Input, Bidirectional
from tensorflow.keras import Model

emb = Input(shape=(35, 200), dtype='float32')
w_c_feature = Bidirectional(LSTM(units=200, return_sequences=True))(emb)

model = Model(emb, w_c_feature)
print(model.summary())