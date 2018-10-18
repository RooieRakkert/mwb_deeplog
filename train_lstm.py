#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2018-10-07 22:58
# * Last modified : 2018-10-16 01:26
# * Filename      : train_lstm.py
# * Description   :
'''
'''
# **********************************************************
# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


filename = "train_log_seqence.txt"
seq_length = 20# l1,l2...l10 -> l_next
n_templates = 14 #模板数量
model_path ='weights/'
template_index_map_path = 'template_to_int.txt'#保存模板号与向量里数值的对应关系

raw_text = []
with open(filename) as IN:
    for line in IN:
        l=line.strip().split()
        raw_text.append(l[1])
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
f = open(template_index_map_path,'w')
for k in char_to_int:
    f.writelines(str(k)+' '+str(char_to_int[k])+'\n')


# summarize the loaded data
n_chars = len(raw_text)
#n_templates = len(char_to_int)
print ("length of log sequence: ", n_chars)
print ("# of templates: ", n_templates)

# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("# of patterns:", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize 我们需要将整数重新缩放到0到1的范围，以使默认情况下使用sigmoid激活函数的LSTM网络更容易学习模式
X = X / float(n_templates)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)#转成one hot,维度为总的tags数量
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = model_path+"log_weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
