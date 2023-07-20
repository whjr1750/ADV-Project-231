import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv(r'books.csv',on_bad_lines ='skip')

x = dataset.iloc[:, [4,11]].values
y = dataset.iloc[:, 3].values

model = Sequential()
model.add(Dense(120, input_dim= 8, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

