from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ctypes import *
import FinanceDataReader as fdr
import datetime
import pandas as pd
import numpy as np
from pytimekr import pytimekr

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

class windowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        L = y.shape[0]
        num_samples = (L - input_window - output_window) // stride + 1

        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y

        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

iw = 24*14
ow = 24*7

df = fdr.DataReader('005380', '2018-05-04', '2023-10-13')

# Data Preprocessing
df['Date'] = pd.to_datetime(df.index)
df = df[['Date', 'Close']]
df = df.set_index('Date')

# Normalize the data
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

train = df[:-24*7]
data_train = train["Close"].to_numpy()

test = df[-24*7:]
data_test = test["Close"].to_numpy()

train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)


class TFModel(nn.Module):
    def __init__(self, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw + ow) // 2),
            nn.ReLU(),
            nn.Linear((iw + ow) // 2, ow)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0, 1), srcmask).transpose(0, 1)
        output = self.linear(output)[:, :, 0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

import __main__
setattr(__main__, "TFModel", TFModel)
setattr(__main__, "PositionalEncoding", PositionalEncoding)
model2 = torch.load("venv/model.h",map_location=torch.device("cpu"))
model3 = torch.load("venv/model2.h",map_location=torch.device("cpu"))
device = torch.device("cpu")

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask

def evaluate():
    input = torch.tensor(data_train[-24*7*2:]).reshape(1,-1,1).to(device).float().to(device)
    model2.eval()

    src_mask = model2.generate_square_subsequent_mask(input.shape[1]).to(device)
    predictions = model2(input, src_mask)
    return predictions.detach().cpu().numpy()

result = evaluate()
result = scaler.inverse_transform(result)[0]

df2 = fdr.DataReader('005930', '2018-05-04', '2023-10-13')

# Data Preprocessing
df2['Date'] = pd.to_datetime(df2.index)
df2 = df2[['Date', 'Close']]
df2 = df2.set_index('Date')

# Normalize the data
scaler = MinMaxScaler()
df2['Close'] = scaler.fit_transform(df2[['Close']])

train2 = df2[:-24*7]
data_train2 = train2["Close"].to_numpy()

test2 = df2[-24*7:]
data_test2 = test2["Close"].to_numpy()

train_dataset2 = windowDataset(data_train2, input_window=iw, output_window=ow, stride=1)
train_loader2 = DataLoader(train_dataset2, batch_size=64)

def evaluate2():
    input = torch.tensor(data_train2[-24*7*2:]).reshape(1,-1,1).to(device).float().to(device)
    model3.eval()

    src_mask = model3.generate_square_subsequent_mask(input.shape[1]).to(device)
    predictions = model3(input, src_mask)
    return predictions.detach().cpu().numpy()

result2 = evaluate2()
result2 = scaler.inverse_transform(result2)[0]

from tensorflow import keras

ldf = fdr.DataReader('005930', '2018-05-04', '2023-10-13')
def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema']-dataset['26ema']

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(window = 21).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()

    return dataset

ldf = get_technical_indicators(ldf)
ldf=ldf.dropna()
ldf=ldf[['Volume',"Close","Change","ma7","ma21","MACD","20sd","upper_band","lower_band","ema"]]
scalar = MinMaxScaler()

dfx = ldf[['Volume',"Close","Change","ma7","ma21","MACD","20sd","upper_band","lower_band","ema"]]
scaled_df = scalar.fit_transform(dfx)

dfy1=[]
for i in range(len(scaled_df)):
  dfy1.append(scaled_df[i][1])
window_size = 10

data_X = []
data_y = []
for i in range(len(dfy1) - window_size):
    _X = scaled_df[i : i + window_size]
    _y = dfy1[i + window_size]
    data_X.append(_X)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])

test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])

lstm2=keras.models.load_model("venv/lstm2.h5")
lstm2.fit(test_X,test_y)
pred_y1=lstm2.predict(test_X)

df = fdr.DataReader('005380', '2018-05-04', '2023-10-13')
df = get_technical_indicators(df)
df=df.dropna()
df=df[['Volume',"Close","Change","ma7","ma21","MACD","20sd","upper_band","lower_band","ema"]]
scalar = MinMaxScaler()

dfx = df[['Volume',"Close","Change","ma7","ma21","MACD","20sd","upper_band","lower_band","ema"]]
scaled_df = scalar.fit_transform(dfx)

dfy=[]
for i in range(len(scaled_df)):
  dfy.append(scaled_df[i][1])
window_size = 10

data_X = []
data_y = []
for i in range(len(dfy) - window_size):
    _X = scaled_df[i : i + window_size]
    _y = dfy[i + window_size]
    data_X.append(_X)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])

test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])

lstm2=keras.models.load_model("venv/lstm2.h5")
lstm2.fit(test_X,test_y)
pred_y=lstm2.predict(test_X)
print(df.Close[-2]*pred_y[-2]/dfy[-2])
print(ldf.Close[-2]*pred_y1[-2]/dfy1[-2])
holidays=pytimekr.holidays()

@app.get("/{num}/{date}/{model}")
async def main(num:int,date:datetime.date, model:int):

    dif=datetime.datetime(2023,10,13).date()-date
    s=int(str(dif).split(" ")[0])
    temp=date+datetime.timedelta(days=1)
    cnt=0
    while temp!=datetime.datetime(2023,10,13).date():
        if temp in holidays or temp.weekday()>4:
            pass
        else:
            cnt+=1
        temp = temp + datetime.timedelta(days=1)
    cnt=-cnt-1
    print(cnt)
    tomorrow=date+datetime.timedelta(days=1)
    temp2=date+datetime.timedelta(days=1)
    while True:
        if temp2 in holidays or temp2.weekday()>4:
            pass
        else:
            break

    if num==1:
        df = fdr.DataReader('005930', temp2, temp2)
        if model==1:
            data = {
                "price": int(ldf.Close[cnt]*pred_y1[cnt]/dfy1[cnt]),
                "real": int(df["Close"][0])
            }
        else:
            data={
                "price":int(result2[cnt]),
                "real":int(df["Close"][0])
            }
    elif num==2:
        df = fdr.DataReader('005380', temp2, temp2)
        if model==1:
            data = {
                "price": int(df.Close[cnt]*pred_y[cnt]/dfy[cnt]),
                "real": int(df["Close"][0])
            }
        else:
            data = {
                "price": int(result[cnt]),
                "real": int(df["Close"][0])
            }

    return data