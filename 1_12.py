import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import cycle
from sklearn import svm
from sklearn.metrics import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import *
from pybrain.structure.modules import *
from pybrain.datasets import *
#from pybrain.datasets.supervised.SequentialDataSet import *



#import quandl


#quandl.ApiConfig.api_key = 'ixDYwP7yyyeazZAKfTP3' 


#static company data,deleted old unrefreshed data
#df = pd.read_csv("NSE-datasets-codes.csv")
#codes = df.iloc[:,0]
#names = df.iloc[:,1]
#print len(codes), codes[(len(codes)-1)]

c = []
#for i in range(0,len(codes)):
for i in range(2):
	c.append("out" + str(i) +".csv")
	#a = quandl.get(codes[i], authtoken="ixDYwP7yyyeazZAKfTP3")
	#a.to_csv(c[i-1],sep='\t')
	#b = a.reset_index().values
	#print codes[i],i, a.tail(1)
	#print b.shape, type(b)
	#data = a.as_matrix()
	#print data.shape
	#print data, type(data)
	df = pd.read_csv(c[i],sep='\t')
	#print df.tail()
	


	#for binary prediction of up and down
	def shifted_y(data):
		y = data[1:]
		y_avg= []
		#predicted y is y after 5 days hence avg y is y in those 5 days(assumption)
		#y_curr_avg = (y[len(y)-1]+y[len(y)-2]+y[len(y)-3]+y[len(y)-4]+y[len(y)-5])/5
		print len(y), len(data) #check@@@@@@@@@@@@@@@@@@@

		'''for i in range(len(y)):
			if (y[i]>):
				y[i]=1
			else:
				y[i]=0'''
		return y

	#increase x data is dataset(date,open,high,low,last,close,total trade,turnover)
	#in panda frame eg: open --->  a.iloc[:,1]
	def extra_x(data,x):
		open_change_percent = []
		close_change_percent =[]
		low_change_percent = []
		high_change_percent = []
		turnover_change_percent = []
		volume_change_percent = []
		volume_diff_percent = []
		open_diff_percent = []
		open_price_moving_avg = []
		close_price_moving_avg= []
		high_price_moving_avg = []
		low_price_moving_avg = []

		highest_open_price = data.iloc[0,1]
		lowest_open_price = data.iloc[0,1]
		highest_volume = data.iloc[0,6]
		lowest_volume = data.iloc[0,6]

		if (x>len(data.iloc[:,1])):
			x=len(data.iloc[:,1])   #x kaun hai bhai @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
		for i in range(len(data.iloc[:,5])-x,len(data.iloc[:,5])):
			if(highest_open_price<data.iloc[i,1]):
				highest_open_price=data.iloc[i,1]
			if(lowest_open_price>data.iloc[i,1]):
				lowest_open_price=data.iloc[i,1]
			if(highest_volume<data.iloc[i,6]):
				highest_volume = data.iloc[i,6]
			if(lowest_volume>data.iloc[i,6]):
				lowest_volume = data.iloc[i,6]

		opensum = data.iloc[0,1]
		closesum = data.iloc[0,6]
		highsum = data.iloc[0,2]
		lowsum = data.iloc[0,3]
		for i in range(1,len(data.iloc[:,2])-1):
			close_change_per = (data.iloc[i,5]-data.iloc[i-1,5])/data.iloc[i+1,5]*100
			close_change_percent.append(close_change_per)

			open_change_per = (data.iloc[i,1]-data.iloc[i-1,1])/data.iloc[i+1,1]*100
			open_change_percent.append(open_change_per)

			high_change_per = (data.iloc[i,2]-data.iloc[i-1,2])/data.iloc[i+1,2]*100
			high_change_percent.append(high_change_per)
			if data.iloc[i-1,6] == 0:
				data.iloc[i-1,6] = data.iloc[i-2,6]

			volume_change_per = (data.iloc[i,6]-data.iloc[i-1,6])/data.iloc[i+1,6]*100
			volume_change_percent.append(volume_change_per)

			low_change_per = (data.iloc[i,3]-data.iloc[i-1,3])/data.iloc[i+1,3]*100
			low_change_percent.append(low_change_per)

			turnover_change_per = (data.iloc[i,7]-data.iloc[i-1,7])/data.iloc[i+1,7]*100
			turnover_change_percent.append(turnover_change_per)

			volume_diff = (data.iloc[i,6]-data.iloc[i-1,6])/(highest_open_price - lowest_open_price)
			volume_diff_percent.append(volume_diff)

			open_diff = (data.iloc[i,1]-data.iloc[i-1,1])/(highest_open_price - lowest_open_price)
			open_diff_percent.append(open_diff)

			opensum = opensum + data.iloc[i,1]
			closesum = closesum + data.iloc[i,5]
			highsum = highsum + data.iloc[i,2]
			lowsum = lowsum + data.iloc[i,3]

			open_price_moving = float(opensum/i+1)/data.iloc[i+1,1]
			open_price_moving_avg.append(open_price_moving)	

			close_price_moving = float(closesum/i+1)/data.iloc[i+1,5]
			close_price_moving_avg.append(close_price_moving)

			high_price_moving = float(highsum/i+1)/data.iloc[i+1,2]
			high_price_moving_avg.append(high_price_moving)

			low_price_moving = float(lowsum/i+1)/data.iloc[i+1,3]
			low_price_moving_avg.append(low_price_moving)


		#converting data to pandas frame
		open_cp = pd.DataFrame({'open change precentage': open_change_percent})
		close_cp =pd.DataFrame({'close change precentage': close_change_percent})
		low_cp = pd.DataFrame({'high change precentage': high_change_percent})
		high_cp = pd.DataFrame({'low change precentage': low_change_percent})
		turnover_cp = pd.DataFrame({'turnover_change_percent': turnover_change_percent})
		volume_cp = pd.DataFrame({'volume_change_percent': volume_change_percent})
		volume_dp = pd.DataFrame({'volume_diff_percentt': volume_diff_percent})
		open_dp = pd.DataFrame({'open_diff_percent': open_diff_percent})
		open_pa = pd.DataFrame({'open_price_moving_avg': open_price_moving_avg})
		close_pa= pd.DataFrame({'close_price_moving_avg': close_price_moving_avg})
		high_pa = pd.DataFrame({'high_price_moving_avg': high_price_moving_avg})
		low_pa = pd.DataFrame({'low_price_moving_avg': low_price_moving_avg})

		#concatinate the features
		frames = [data,open_cp,close_cp,low_cp,high_cp,turnover_cp,volume_cp,volume_dp,open_dp,open_pa,close_pa,high_pa,low_pa]
		result = pd.concat(frames, axis=1)
		#data = data.join(result)
		return result


	def train(x,y):
		x = x.drop(x.columns[[0]],axis=1)
		y = y.drop(y.columns[[0]],axis=1)

		len_x = len(x)
		len_train = int(0.75*len_x)
		train_x = x[0:len_train].as_matrix()
		test_x = x[len_train:].as_matrix()

		train_y = y[0:len_train].as_matrix()
		test_y = y[len_train:].as_matrix()
		'''f = svm.SVC(C=100000,kernel = 'rbf',gamma=0.001)

		#convert to numpy befour fitting
		train_x1 = train_x.drop(train_x.columns[[0]],axis=1).as_matrix()
		#print train_x1
		train_y1 = train_y.as_matrix()
		print train_y1
		test_x1 = test_x.drop(test_x.columns[[0]],axis=1).as_matrix()
		y1 = y.as_matrix()

		#fit to model
		print "@@@@@@@@@@@@@@@", train_x1.shape,"AAAAA",train_y1.shape
		f.fit(train_x1,train_y1)
		print type(f)

		#predict test y
		predicted = f.predict(test_x1)
		print "acuracy:" , accuracy(y,predicted)'''
		return test_x,test_y, train_x, train_y





	def nn(train_x,train_y,test_x,test_y):
		print train_x.shape,train_y.shape


		net = buildNetwork(1,5,1,hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
		ds = SequentialDataSet(1,1)
		

		for i,j in zip(train_x,cycle(train_x[1:])):
			ds.addSample(i,j)
		trainer = BackpropTrainer(net,ds)
		#print trainer

		epochs = 5
		cyc = 100
		epochs_n = 500
		train_error=[]
		ty = []
		for i in range(cyc):
			trainer.trainEpochs(epochs)
			train_error.append(trainer.testOnData())
			epochs_n = (i+1)*epochs
		#print train_error, "train error"

		predicte = list()
		for i,j in ds.getSequenceIterator(0):
			predicte.append(net.activate(i))
			ty.append(j)
		predicted = np.array(predicte)
		test_y = np.array(ty)
		#print predicted , test_y
		print "acuracy", accuracy(test_y,predicted)
		return predicted

	def accuracy(y,pred_y):
		mse_y = (y-pred_y)
		mse_y1 = np.sum(mse_y,axis=0)/19
		return mse_y1 

data1 = extra_x(df,5)
#print data1.tail()
y1= shifted_y(data1)

#data and output to numpy
test_x1,test_y1, train_x1, train_y1 = train(data1,y1)
#print train_y1,train_x1

predicted_y = np.zeros(shape=19)
#print predicted_y.shape
#train on rnn
for i in range(19):
	print i
	predicted = nn(train_x1[i], train_y1[i] , test_x1[i],test_y1[i])	








