#runing in each problem 
#EX: A1 means runing in problem 1
# and accurarcy_test means runing A1 in accuracy test function so does accuracy_train

import numpy as np
from numpy import linalg as LA
import mnist_reader
import matplotlib.pyplot as plt
X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

tau=1;
yhat=0;
y_label=[];
percepton_mistake=[]
w=np.zeros(28*28)
w_mulit=np.ones(28*28*10)
weight=[];
#	np.argmax(yhat_multi) -> yhat
def PA_multi_perceptron():
	global w_mulit;
	mistake=0;
	np.set_printoptions(threshold=np.inf)
	fx=[]
	temprep=[]
	for i in range(0, len(X_train)):
		yhat=(np.argmax((rep_fx(i,w_mulit)[0])))
		if yhat!=y_train[i]:
			t=(1-w_mulit*(rep_fx(i,0)[1][y_train[i]])-w_mulit*rep_fx(i,0)[1][yhat])/(LA.norm(rep_fx(i,0)[1][y_train[i]]-rep_fx(i,0)[1][yhat]))**2
			w_mulit=w_mulit+(t*rep_fx(i,0)[1][y_train[i]]-rep_fx(i,0)[1][yhat])
			print (mistake)
			mistake =mistake+1;
		weight.append(w_mulit)
	return mistake
def multi_perceptron():
	global w_mulit;
	mistake=0;
	fx=[]
	temprep=[]
	for i in range(0, len(X_train)):
		yhat=(np.argmax((rep_fx(i,w_mulit)[0])))
		if yhat!=y_train[i]:
			w_mulit=w_mulit+(1*rep_fx(i,0)[1][y_train[i]]-rep_fx(i,0)[1][yhat])
			mistake =mistake+1;
			print (mistake)
		#weight.append(w_mulit)
	return mistake
def average_multi_perceptron( k):
	global w_mulit;
	mistake=0;
	fx=[]
	c=1;
	u=np.zeros(28*28*10)
	temprep=[]
	if k >0:
		w_mulit=weight[k-1]
	for i in range(0, len(X_train)):
		yhat=(np.argmax((rep_fx(i,w_mulit)[0])))
		if yhat!=y_train[i]:
			w_mulit=w_mulit+(1*rep_fx(i,0)[1][y_train[i]]-rep_fx(i,0)[1][yhat])
			u=u+y_train[i]*c*(rep_fx(i,0)[1][y_train[i]]-rep_fx(i,0)[1][yhat])
			mistake =mistake+1;
			print(mistake)
		c=c+1;
		weight.append(w_mulit-(1/c)*u)
	return mistake
def rep_fx(x ,v):
	j=0;
	q=0;
	newarr=[]
	g=[]
	for i in range(0,10):
		q=1;
		temp=np.zeros(28*28*10)
		for j in range (28*28*i,28*28*i+784):
			temp[j]=X_train[x][q]
			if(q<783):
				q=q+1;
		g.append(temp)
		newarr.append(np.dot(temp,v))
	return newarr ,g
def accuracyrate_train():
	del y_label[:]
	classifylabel(y_train);
	print("Accuracy")
	arr_accuracy=[];
	accuracy=0;
	for i in range(0, 20):
		accuracy=0;
		for j in range(0,len(y_train)):
			yhat=(np.dot(weight[i],X_train[j]))
			if sign(yhat)==y_label[j]:
				accuracy=accuracy+1;
		arr_accuracy.append(accuracy/len(y_train))
	print(arr_accuracy)
	plt.xlabel("Train dataset")
	plt.xlabel("iteration")
	plt.ylabel("Accuracy")
	plt.plot(range(0, 20),arr_accuracy)
	plt.show()
def accuracyrate_test():
	del y_label[:]
	classifylabel(y_test);
	print("Accuracy")
	arr_accuracy=[];
	accuracy=0;
	for i in range(0, 20):
		accuracy=0;
		for j in range(0,len(y_test)):
			yhat=(np.dot(weight[i],X_test[j]))
			if sign(yhat)==y_label[j]:
				accuracy=accuracy+1;
		arr_accuracy.append(accuracy/len(y_test))
	print(arr_accuracy)
	plt.xlabel("Testing dataset")
	plt.xlabel("iteration")
	plt.ylabel("Accuracy")
	plt.plot(range(0, 20),arr_accuracy)
	plt.show()

def perceptron_D( val):
	mistake=0;
	if val> 60000:
		val=60000
	global w;
	for i in range(0,val):
		yhat=(np.dot(w,X_train[i]))
		if sign(yhat)!=y_label[i]:
			w=w+1*np.dot(y_label[i],X_train[i]);
			mistake=mistake+1;
	weight.append(w);
	return mistake;
def B_1():
	for i in range(0,50):
		percepton_mistake.append(multi_perceptron());
		print(percepton_mistake[i])
def B_2():
	for i in range(0,20):
		percepton_mistake.append(PA_multi_perceptron());
		print(percepton_mistake[i],percepton_mistake[i]/60000)
def B_3():
	for i in range(0,1):
		percepton_mistake.append(average_multi_perceptron(i));
		print(percepton_mistake[i],percepton_mistake[i]/60000)

print( "#runing in each problem EX: A1 means runing in problem 1 and accurarcy_test means runing A1 in accuracy test function so does accuracy_train")
B_1();
#B_2();
#B_3();
#B_4();
accuracyrate_test()
#accuracyrate_train();
