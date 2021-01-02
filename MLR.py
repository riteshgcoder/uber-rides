import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn
data=pd.read_csv(r'C:\Github repo\uber ride prediction\taxi.csv')
data_x=data.iloc[:,0:-1].values
data_y=data.iloc[:,-1].values
#perforn train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3,random_state=0)
# applying alg
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
print("Train score:",reg.score(x_train,y_train))#train score
print("Train score:",reg.score(x_test,y_test))
pickle.dump(reg,open('taxi.pkl','wb'))
model=pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,2000000,727,22]]))
