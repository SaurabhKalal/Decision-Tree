import pandas as pd
data=pd.read_csv("Social_Network_Ads.csv")
print(data.isna().sum())
X=data.iloc[:,1:4].values
Y=data.iloc[:,-1].values
Y=Y.reshape(Y.shape[0],1)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),[0])],remainder='passthrough')
X=CT.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc=sc.fit(X)
X=sc.transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)
Max_depth=[]
from sklearn.tree import DecisionTreeClassifier
for i in range(1,10):
    model=DecisionTreeClassifier(max_depth=i)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(Y_test, Y_pred)
    print(cm)
    from sklearn.metrics import accuracy_score
    score=accuracy_score(Y_test,Y_pred)
    print(score)
    Max_depth.append(score)

import matplotlib.pyplot as plt
plt.plot(range(1,10),Max_depth)
plt.title("Accuracy plot of Decision tree model")
plt.xlabel("Depth values")
plt.ylabel("Accuracy Score")
plt.show()


