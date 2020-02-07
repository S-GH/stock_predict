# 삼성전자 LSTM
# output = RMSE + 주가
import pandas as pd
import numpy as np

# 시계열 데이터 전처리
def split_sequence(sequence,n_steps):
    X,y = list(),list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix],sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

sam = pd.read_csv('data/samsung.csv',encoding='euc-kr')
sam = sam.sort_values(['일자'],ascending=[True])
del sam['일자']
sam_ndarr = sam.values

# 56,000 (str) > 56000 (int) convert
lst2 = []
for  arr in sam_ndarr:
    lst1 = []
    for a in arr:
        a = a.replace(",","")
        lst1.append(int(a))
    lst2.append(lst1)
# print(np.array(lst2).shape)

x_data,y_data = split_sequence(lst2,5)
x_data = np.array([x[3] for x in x_data])
y_data = np.array([x[3] for x in y_data])
y_data = y_data.reshape(y_data.shape[0],1)
print(x_data.shape)
# print(y_data.shape)
# x_data = x_data.reshape(x_data.shape[0],5)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# StandardScaler 스칼라
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_data = x_data.reshape(x_data.shape[0],5,1)
print(x_data.shape)

# train, test split
from sklearn.model_selection import train_test_split
x_train, x_test,  y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.7, random_state=66, shuffle = False)

x_test, x_val,  y_test, y_val = train_test_split(
    x_test, y_test, test_size=0.5, random_state=66, shuffle = False)

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout

input1 = Input(shape=(5,1))
LSTM1 = LSTM(64,return_sequences=True, activation='relu')(input1)
DP1 = Dropout(0.2)(LSTM1)
LSTM2 = LSTM(64,return_sequences=True, activation='relu')(DP1)
DP2 = Dropout(0.2)(LSTM2)
LSTM3 = LSTM(64, activation='relu')(DP2)
DP3 = Dropout(0.2)(LSTM3)
output = Dense(1)(DP3)

model = Model(inputs=input1, outputs=output)
model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs = 1000, batch_size=32, validation_data= (x_val, y_val))

#4. 평가
loss, mse = model.evaluate(x_test, y_test, batch_size=5)
print('mse: ', mse)

y_predict = model.predict(x_test,batch_size=5)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ",RMSE(y_test,y_predict))

x_prd = x_data[-1]
x_prd = x_prd.reshape(1,5,1)
aaa= model.predict(x_prd, batch_size=1)
print('2/3일 예측값 = ',aaa)

# 히스토리 시각화.
# print(history.history.keys())
# val_acc = history.history['val_loss']
# acc = history.history['loss']

# import matplotlib.pyplot as plt

# x_len = np.arange(len(acc))
# plt.plot(x_len, acc, marker='.', c='blue', label="Train-set mse.")
# plt.plot(x_len, val_acc, marker='.', c='red', label="Validation-set mse.")

# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('mse')
# plt.show()