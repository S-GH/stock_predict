# 삼성전자 + kospi ensemble DNN
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
kospi = pd.read_csv('data/kospi200.csv',encoding='euc-kr')
del sam['일자']
del kospi['일자']

t = kospi['거래량'].to_numpy()
klst = []
for k in t:
    k = k.replace(",","")
    klst.append(int(k))
kospi['거래량'] = klst

sam_ndarr = sam.values
kospi_ndarr = kospi.values

# 56,000 (str) > 56000 (int) convert
slst2 = []
for sarr in sam_ndarr:
    slst1 = []
    for s in sarr:
        s = s.replace(",","")
        slst1.append(int(s))
    slst2.append(slst1)
    
s_x_data, y_data = split_sequence(slst2,1)
k_x_data, _ = split_sequence(kospi_ndarr,1)
print(s_x_data.shape,y_data.shape)
print(k_x_data.shape)
# (425 ,1 ,5) > (425,5)

s_x_data = s_x_data.reshape(s_x_data.shape[0],5)
k_x_data = k_x_data.reshape(k_x_data.shape[0],5)
print(s_x_data.shape,s_x_data.shape)
print(k_x_data.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# min max 스칼라
scaler = MinMaxScaler()
scaler.fit(s_x_data)
s_x_data = scaler.transform(s_x_data)
k_x_data = scaler.transform(k_x_data)

# train, test split
from sklearn.model_selection import train_test_split
s_x_train, s_x_test, k_x_train, k_x_test,  y_train, y_test = train_test_split(
    s_x_data, k_x_data, y_data, train_size=0.6, random_state=66, shuffle = False)
    
s_x_test, s_x_val, k_x_test, k_x_val,  y_test, y_val = train_test_split(
    s_x_test, k_x_test, y_test, test_size=0.5, random_state=66, shuffle = False)

print(s_x_train.shape,s_x_test.shape,s_x_val.shape)
print(k_x_train.shape,k_x_train.shape,k_x_train.shape)
print(y_train.shape,y_test.shape,y_val.shape)


from keras.models import Sequential, Model
from keras.layers import Dense, Input

# Model 1
input1 = Input(shape=(5,))
dense11 = Dense(5,activation='relu')(input1)
dense12 = Dense(2)(dense11)
dense13 = Dense(3)(dense12)
output1 = Dense(1)(dense13)

# Model 2
input2 = Input(shape=(5,))
dense21 = Dense(5,activation='relu')(input2)
dense22 = Dense(2)(dense21)
dense23 = Dense(3)(dense22)
output2 = Dense(1)(dense23)

from keras.layers.merge import concatenate
# axis 주의 !!
merge1 = concatenate([output1,output2])

# Model 3
middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(5)(middle2)

model = Model(inputs=[input1,input2], outputs=output)
model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit([s_x_train,k_x_train], y_train, epochs =100, batch_size=1, validation_data= ([s_x_val,k_x_val], y_val))

#4. 평가
loss, mse = model.evaluate([s_x_test,k_x_test], y_test, batch_size=1)
print('mse: ', mse)

y_predict = model.predict([s_x_test,k_x_test],batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ",RMSE(y_test,y_predict))

s_x_prd = np.array(slst2[-1])
s_x_prd = s_x_prd.reshape(1,5)

k_x_prd = kospi_ndarr[-1]
k_x_prd = k_x_prd.reshape(1,5)
aaa = model.predict([s_x_prd,k_x_prd], batch_size=1)
print('2/3일 예측값 = ',aaa)