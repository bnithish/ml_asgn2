import numpy as np
from utils import loadData

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
model = Sequential()
model.add(Dense(units=64, activation='relu',input_dim=225))
model.add(Dense(units=1, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

x,y = loadData("train",225)
x = x.toarray()
train_x = x[0:10000]
train_y = y[0:10000]
y_binary = to_categorical(train_y)
test_x = x[9000:10000]
test_y = y[9000:10000]
model.fit(train_x,y_binary,epochs = 5)
result = model.predict(test_x)
print(result)
