import pandas as pd
from matplotlib import pyplot as plt


df=pd.read_csv("insurance_data.csv")
df.head()

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')


from sklearn.model_selectionaction import train_test_split

X_train ,X_test,Y_train ,Y_test  =train_test_split(df[['age']],df.bought_insurance,test_size=0.1)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)

y_predicted=model.predict(X_test)

model.score(Y_test,y_predicted)