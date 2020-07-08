import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#Cay quyet dinh
data = pd.read_csv('zoo.csv')
#print (data)
x = data.drop(columns=['animal name','type'])
#print (x)
y = data.type
#print (y)
table=pd.crosstab(x.backbone,y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Tương quan giữa loại động vật và động vật có sương sống')
plt.xlabel('ĐV có sương sống')
plt.ylabel('Loài')
plt.savefig('mariral_vs_pur_stack')
plt.show()
'''
#bieu do cot the hien so luong dong vat theo tung loai
tenloai=['ĐV có vú','Chim','ĐV dưới nước','Lưỡng cư','Bò sát','Côn trùng','Không xương sống']
soluong=[41,20,13,4,5,8,10]
plt.bar('ĐV có vú',41,color='red')
plt.bar('Chim',20,color='yellow')
plt.bar('ĐV dưới nước',13,color='blue')
plt.bar('Lưỡng Cư',4,color='pink')
plt.bar('Không SX',10,color='green')
plt.bar('Bò sát',5,color='black')
plt.bar('Côn trùng',8,color='purple')
plt.xlabel('Tên loài')
plt.ylabel('Số lượng')
plt.title('Số lượng của từng loại động vật')
plt.show()
'''
'''
#bieu do tròn bieu hiện so luong cua tung loại ĐV
tenloai=['ĐV có vú','Chim','ĐV dưới nước','Lưỡng cư','Bò sát','Côn trùng','Không xương sống']
soluong=[41,20,13,4,5,8,10]
#Explode=[0,0.1,0,0,0,0]
plt.pie(soluong,labels=tenloai,shadow=True,startangle=45)
plt.axis('equal')
plt.legend(title='Loại Động Vật')
plt.show()
'''

#Su dụng SVM de phan loai

# Sử dụng nghi thức kiểm tra hold-out
# Chia dữ liệu ngẫu nhiên thành 2 tập dữ liệu con:
# training set và test set theo tỷ lệ 70/3
'''
print ("SVM:")
print ("Nghi thức kiểm tra Hold-out")
tb=0
n=20
for i in range(n):
	X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
	model = svm.SVC(kernel='rbf')
	model.fit(X_train,y_train)
	model.fit(X_train,y_train)
	#Du doan nhan tap kiem tra

	y_pred=model.predict(X_test)
	#print (y_pred)

	ac=model.score(X_test,y_test)
	print ("Do chinh xac cua lan thu ", i+1 ,"là: ",ac)	
	tb+=ac
print ("Do chinh xac trung binh mo hinh SVM sd hold-out: ",tb/n)
print ("=================")

#model = svm.SVC(kernel='linear', C=100)
#DCX=0.93
# Xây dựng mô hình svm sử dụng hàm nhân (kernel) là RBF
# SVC là viết tắt của từ Support Vector Classification
model = svm.SVC(kernel='rbf')
scores = cross_val_score(model, x, y, cv=5)
predicted = cross_val_predict(model, x, y, cv=5) 
#model.fit(X_train, y_train)
# Dự đoán nhãn tập kiểm tra
#prediction = model.predict(X_test)
ac_score = metrics.accuracy_score(y, predicted)
print (ac_score)

cl_report = metrics.classification_report(y, predicted)
print (cl_report)
print("Do chinh xac cua mo hinh SVM sd 5-fold" , np.mean(scores))

print ("=================\n\n\n")
print ("=================")
#Giai thuat Knn
#su dung nghi thuc hold-out
print ("K láng giềng:")
print ("Nghi thức kiểm tra Hold-out!")
t=0
n=20
for i in range(n):
	X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
	model = KNeighborsClassifier(n_neighbors=3)
	model.fit(X_train,y_train)
	model.fit(X_train,y_train)
	#Du doan nhan tap kiem tra

	y_pred=model.predict(X_test)
	#print (y_pred)

	ac=model.score(X_test,y_test)
	print ("Do chinh xac cua lan thu ", i+1 ,"là: ",ac)	
	t+=ac
print ("Do chinh xac TB KNN sd hold-out: ",t/n)
print ("=================")
#Kiem tra cheo KNN
print("Kiem tra cheo knn, nFold=5")
model = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(model, x, y, cv=5)
predicted = cross_val_predict(model, x, y, cv=5) 
#model.fit(X_train, y_train)
# Dự đoán nhãn tập kiểm tra
#prediction = model.predict(X_test)
ac_score = metrics.accuracy_score(y, predicted)
print (ac_score)
cl_report = metrics.classification_report(y, predicted)
print (cl_report)
print("Do chinh xac cua mo hinh KNN 5-fold: " , np.mean(scores))

print ("=================\n\n\n")
print ("=================")
#Cây quyết định

from sklearn.model_selection import train_test_split
print ("Cay quyết định:")
print ("Nghi thuc kiem tra Hold-out")
tong=0
n=20
for i in range(n):
	X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
	model = tree.DecisionTreeClassifier(criterion="entropy",min_samples_leaf=5)
	model.fit(X_train,y_train)
	#Du doan nhan tap kiem tra

	y_pred=model.predict(X_test)
	#print (y_pred)

	ac=accuracy_score(y_test,y_pred)
	print ("Do chinh xac cua lan thu ", i+1 ,"là: ",ac)	
	tong+=ac
print ("Do chinh xac TB mh Cay quyet dinh sd hold-out: ",tong/n)
print("Kiem tra cheo cay quyet dinh, nFold=5")
print ("=================")
model = tree.DecisionTreeClassifier(criterion="entropy",min_samples_leaf=5)
scores = cross_val_score(model, x, y, cv=5)
predicted = cross_val_predict(model, x, y, cv=5) 
#model.fit(X_train, y_train)
# Dự đoán nhãn tập kiểm tra
#prediction = model.predict(X_test)
ac_score = metrics.accuracy_score(y, predicted)
print (ac_score)
cl_report = metrics.classification_report(y, predicted)
print (cl_report)
print("Do chinh xac cua mo hinh Cay quyet dinh:" , np.mean(scores))
tree.plot_tree(model.fit(x,y))
DV_moi=([[0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]])
print (model.predict(DV_moi))
from joblib import dump, load
dump(model, 'Zoo.joblib') 

#plt.show()

'''

