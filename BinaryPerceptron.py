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
def classifylabel(dataset):
	for i in range(0, len(dataset)):
		if(dataset[i]%2==0):
			y_label.append(1);
		else:
			y_label.append(-1);
def sign(z):
    if z > 0:
        return 1
    else:
        return -1
def perceptron():
	mistake=0;
	global w;
	for i in range(0,len(X_train)):
		yhat=(np.dot(w,X_train[i]))
		if sign(yhat)!=y_label[i]:
			w=w+1*np.dot(y_label[i],X_train[i]);
			mistake=mistake+1;
	weight.append(w);
	return mistake;
def PA_perceptron():
	mistake=0;
	global w;
	for i in range(0,len(X_train)):
		yhat=(np.dot(w,X_train[i]))
		if sign(yhat)!=y_label[i]:
			t=(1-y_label[i]*yhat)/((LA.norm(X_train[i]))**2)
			w=w+t*np.dot(y_label[i],X_train[i])
			mistake=mistake+1;	
	weight.append(w);
	return mistake;
def average_perceptron():
	classifylabel(y_train)
	mistake=0;
	global w;
	c=1;
	u=np.zeros(28*28)
	for i in range(0,len(X_train)):
		yhat=((np.dot(w,X_train[i])))
		if sign(yhat)!=y_label[i]:
			w=w+np.dot(y_label[i],X_train[i]);
			u=u+np.dot(y_label[i],X_train[i])*c;
			mistake =mistake+1;
		c=c+1;
	w=w-(1/c)*u;
	weight.append(w);
	return mistake
def A_1():
	classifylabel(y_train);
	for i in range(0,50):
		percepton_mistake.append(perceptron());
		print(percepton_mistake[i])
	plt.figure()
	plt.plot(range(len(percepton_mistake)),percepton_mistake)
	plt.title("Percepton")
	plt.xlabel("training iteration")
	plt.ylabel("Mistakes")
	plt.show()
def A_2():
	classifylabel(y_train);
	for i in range(0,50):
		percepton_mistake.append(PA_perceptron());
		print(percepton_mistake[i])
	plt.figure()
	plt.plot(range(len(percepton_mistake)),percepton_mistake)
	plt.title(" PA_percepton")
	plt.xlabel("training iteration")
	plt.ylabel("Mistakes")
	plt.show()
def A_3():
	classifylabel(y_train);
	for i in range(0,50):
		percepton_mistake.append(average_perceptron());
		print(percepton_mistake[i])
	plt.figure()
	plt.plot(range(len(percepton_mistake)),percepton_mistake)
	plt.title(" average_perceptron")
	plt.xlabel("Mistakes")
	plt.ylabel("training iteration")
	plt.show()
def A_4():
	classifylabel(y_train);
	for i in range(0,20):
		percepton_mistake.append(perceptron_D(i*5000));
		print(percepton_mistake[i])
	plt.figure()
	plt.plot(range(len(percepton_mistake)),percepton_mistake)
	plt.title("Percepton")
	plt.xlabel("training iteration")
	plt.ylabel("Mistakes")
	plt.show()
A_3()