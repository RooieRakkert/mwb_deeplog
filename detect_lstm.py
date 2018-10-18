#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2018-10-08 02:24
# * Last modified : 2018-10-16 01:28
# * Filename      : detect_lstm.py
# * Description   :
'''
'''
# **********************************************************
# Load Larger LSTM network and generate text
import sys
import math
from sklearn.metrics import precision_recall_fscore_support
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "test_log_seqence.txt"
seq_length = 10 # l1,l2...l10 -> l_next
n_candidates = 3#top n probability of the next tag
windows_size = 3#hours
step_size = 1#时间窗口的滑动步长，hours
model_filename = "weights/weights-improvement-49-0.4436-bigger.hdf5"#训练好的参数
n_templates = 14 #模板数量，要与train的一致
template_index_map_path = 'template_to_int.txt'#保存模板号与向量里数值的对应关系
label_file = 'labels.txt' #label文件，本样例中从日志文件中抽取label

raw_text = []
raw_time_list = []
raw_label_list = []
with open(filename) as IN:
    for line in IN:
        l=line.strip().split()
        raw_text.append(l[1])
        raw_time_list.append(int(l[0]))

with open(label_file) as IN:
    for line in IN:
        raw_label_list.append(int(line.strip()))

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))

char_to_int = {}
int_to_char = {}
with open(template_index_map_path) as IN:
    for line in IN:
        l = line.strip().split()
        c = l[0]
        i = int(l[1])
        char_to_int[c] = i
        int_to_char[i] = c

# summarize the loaded data
n_chars = len(raw_text)
n_templates = len(chars)
print ("length of log sequence: ", n_chars)
print ("# of templates: ", n_templates)
# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
timeY = []
charX = []
label_list = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    time_out = raw_time_list[i + seq_length]
    label_out = raw_label_list[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    charX.append(seq_in)
    dataY.append(char_to_int[seq_out])
    timeY.append(time_out)
    label_list.append(label_out)
n_patterns = len(dataX)
print ("# of patterns: ", n_patterns)
#split time into windows



# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_templates)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
model.load_weights(model_filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# detect by tag
total=0
anomaly_count_dir = {}
for i in range(n_candidates):
    anomaly_count_dir[i+1] = []

for x_chars,x,aim_y_int in zip(charX,X,dataY):
    total+=1
    air_x_chars = '-'.join(x_chars)
    aim_y_char = int_to_char[aim_y_int]
    #print(air_x_chars,aim_y_char)
    x = numpy.reshape(x, (1, x.shape[0], 1))
    #x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)#输出一个len(tags)的向量，数值越高的列对应概率最高的类别
    index = numpy.argmax(prediction)#argmax返回的是最大数的索引
    result = int_to_char[index]
    #print('input:',air_x_chars,'aim:',aim_y_char,'output:',result)
    prediction=numpy.array(list(prediction)[0])
    for i in range(n_candidates):
        i += 1
        top_n_index=prediction.argsort()[-i:]
        top_n_tag=[int_to_char[index] for index in top_n_index]
        if aim_y_char not in top_n_tag:
            anomaly_count_dir[i].append(1)
        else:
            anomaly_count_dir[i].append(0)


#count by windows
window_count_dir = {}
time_start=timeY[0]
time_end=timeY[-1]
windows_num = max(0, int(((time_end - time_start) - windows_size * 3600) / step_size / 3600)) +  1
windows_start_time = [time_start+i*step_size*3600 for i in range(windows_num)]

raw_windows_label_list = numpy.zeros(windows_num)
for i in range(n_candidates):
    i += 1
    window_count_dir[i] = numpy.zeros(windows_num)
    for cur_time,cur_flag,label in zip(timeY,anomaly_count_dir[i],label_list):
        cur_index = int(max(0,cur_time - time_start - windows_size * 3600) / step_size / 3600)
        window_count_dir[i][cur_index] += cur_flag
        raw_windows_label_list[cur_index] += label
windows_label_list = [ 1 if n >=1 else 0 for n in raw_windows_label_list]


'''
precision, recall, f1_score, _ = np.array(list(precision_recall_fscore_support(testing_labels, prediction)))[:, 1]
print('=' * 20, 'RESULT', '=' * 20)
print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))
'''



print('\nanomaly detection result:')
for i in range(n_candidates):
    i += 1
    print('next tag  is not in top'+str(i)+' candidates:')
    print('# of anomalous/total logs:',str(sum(anomaly_count_dir[i]))+'/'+str(len(anomaly_count_dir[i])))

    precision, recall, f1_score, _ = numpy.array(list(precision_recall_fscore_support(label_list, anomaly_count_dir[i])))[:, 1]
    #print('=' * 20, 'RESULT', '=' * 20)
    print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))

    windows_results = [ 1 if n >=1 else 0 for n in window_count_dir[i]]
    print('# of anomalous/total windows:',str(sum(windows_results))+'/'+str(len(windows_results)))
    precision, recall, f1_score, _ = numpy.array(list(precision_recall_fscore_support(windows_label_list,           windows_results)))[:, 1]
    print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))
    print('')



print ("\nDone.")





