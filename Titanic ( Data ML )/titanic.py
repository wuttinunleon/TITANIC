import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

columns_target = ['Survived']
columns_train = ['Age','Pclass','Sex','Fare','SibSp','Parch','Cabin']

x = train[columns_train]
y = train[columns_target]

x_t = test[columns_train]
x['Cabin']=x['Cabin'].fillna(0)

for i,c in enumerate(x['Cabin']):
    if x['Cabin'][i] != None:
        x['Cabin'][i] =1
        
x['Age']=x['Age'].fillna(x['Age'].median())
x_t['Age']=x_t['Age'].fillna(x_t['Age'].median())
x_t['Fare']=x_t['Fare'].fillna(x_t['Fare'].median())

pValue_sex = {'male':0,'female':1}

x['Sex'] = x['Sex'].apply(lambda x:pValue_sex[x])
x_t['Sex'] = x_t['Sex'].apply(lambda x_t:pValue_sex[x_t])
#print(x_t[x_t.Fare.isnull()])

newC=0.1
newAcc = 0
   # x_train,x_test,y_train,y_test = train_test_split(x[:500],y[:500],random_state=1)
while  newC < 10**4 :
    clf = svm.LinearSVC(C=newC,random_state=1)
    clf.fit(x[:500],y[:500])
    pred = clf.predict(x[500:])
    acc = accuracy_score(pred,y[500:])
    if acc>newAcc:
        newAcc= acc
        print(newAcc,"hi",newC)
        newC+=1
print(acc,newAcc,newC)
 #   print(newAcc,newC)
#    newC+=0.1        
"""
write = pd.DataFrame(pred,test.PassengerId)
write.to_csv('final.csv')

plt.hist(pred)
"""








