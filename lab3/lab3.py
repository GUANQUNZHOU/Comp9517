import numpy as np
import cv2
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dic_num = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
def knn(document,dict):
	G = []
	k = []
	p = os.listdir(r"./"+document)
	p.sort()
	for i in p:
		n = re.search('(.+?)_',i)
		k.append(dict[n.group(1)])
	k = np.array(k)
	for f in p:
		name = document+"/"+f
		img = cv2.imread(name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		G.append(cv2.resize(gray,(32,32)).flatten())
	G = np.array(G)

	train,test,train_l,test_l = train_test_split(G,k,test_size=0.2, random_state=42)
	neigh = KNeighborsClassifier(n_neighbors=5)
	neigh.fit(train, train_l)
	pred = neigh.predict(test)
	accuracy = neigh.score(test,test_l)
	print('The accuracy is {:.2}'.format(accuracy))
	cm = confusion_matrix(test_l,pred)
	print(cm)
	cr = classification_report(test_l, pred)
	print(cr)
knn('data',dic_num)