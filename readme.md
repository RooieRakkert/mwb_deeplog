

## 训练：train_lstm.py
	输入文件：
		train_log_sequence.txt  格式： timestamp + tag
	输出文件:
		template_to_int.txt
		weights/weights-improvement-**-**-bigger.hdf5 有多个文件，选择一个loss值最小的文件，其他可以删掉

## 检测:detect_lstm.py
	输入文件:
		template_to_int.txt  是训练文件生成的
		test_log_sequence.txt  格式： timestamp + tag
		labels.txt   用于评价准确率
		weights/weights-improvement-**-**-bigger.hdf5 loss值最小的文件

	输出说明：
		例如：anomaly detection result:
		next tag  is not in top1 candidates:
		# of anomalous/total logs: 963/1990
		Precision:  0.131880, Recall: 0.900709, F1_score: 0.230072
		# of anomalous/total windows: 349/5096
		Precision:  0.108883, Recall: 0.826087, F1_score: 0.192405

		next tag  is not in top2 candidates:
		# of anomalous/total logs: 639/1990
		Precision:  0.184664, Recall: 0.836879, F1_score: 0.302564
		# of anomalous/total windows: 285/5096
		Precision:  0.122807, Recall: 0.760870, F1_score: 0.211480

		next tag  is not in top3 candidates:
		# of anomalous/total logs: 387/1990
		Precision:  0.108527, Recall: 0.297872, F1_score: 0.159091
		# of anomalous/total windows: 229/5096
		Precision:  0.113537, Recall: 0.565217, F1_score: 0.189091

		top1，top2指的是当下一个tag不是预测概率最大的topn时，认为该点为异常。
		logs和windows分别表示的是日志级和窗口级的异常。
		
本代码是我根据自己的理解，复现自[Du M, Li F, Zheng G, et al. DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning[C]// ACM Sigsac Conference on Computer and Communications Security. ACM, 2017:1285-1298.](http://www.flux.utah.edu/paper/261)	