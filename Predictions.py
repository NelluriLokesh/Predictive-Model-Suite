import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

## Predefiend dataset
from sklearn.datasets import load_diabetes
## Machine Learning librarys
## to split the data into 2 sets :
from sklearn.model_selection import train_test_split

## cheack the performance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def model(x,y,n):
    def decision_tree(x,y,n):
        print("")
        print("")
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

        # for model selection
        from sklearn.tree import DecisionTreeRegressor
        lrmodel = DecisionTreeRegressor()

        # Training the model
        lrmodel.fit(x_train, y_train)
        
        # print(lrmodel.predict([[n]]))
        
        pred = lrmodel.predict(x_test)
        
        dt = r2_score(y_test,pred)
        print("Decision Tree : ")
        print("Accuracy of Decision tree Regression :",dt)
        print("")
        print("The predict value is : ",lrmodel.predict([[n]]))

        # the mean_absolute_error values
        mae = mean_absolute_error(y_test,pred)
        # the mean_squared_error values
        mse = mean_squared_error(y_test, pred)
        print("")
        print("the Mean absolute error : ",mae)
        print("")
        print("the Mean squared error : ",mse)

    def linear_regression(x,y,n):
        x = x.reshape(-1,1)
        y = y

        x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

        from sklearn.linear_model import LinearRegression
        model = LinearRegression() # creating object from class linear regression

        ## Fitting the model/ train the model
        model.fit(x_train , y_train)
        pred = model.predict(x_train)
        
        lr = r2_score(y_train,pred)
        print("")
        print("Linear Regression : ")
        print("Accuracy of Linear Regression :",lr)
        print("The predict value is : ",model.predict([[n]]))
        
        # the mean_absolute_error values
        mae = mean_absolute_error(y_train,pred)
        # the mean_squared_error values
        mse = mean_squared_error(y_train, pred)
        print("")
        print("the Mean absolute error : ",mae)
        print("")
        print("the Mean squared error : ",mse)
        
    def Random_forest_regression(x,y,n):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()

        n_estimators = [int(i) for i in np.linspace(start=100, stop=1200, num=12)]
        max_feature = ['auto','sqrt']
        max_depth = [int(i) for i in np.linspace(start=5, stop=30, num=6)]
        min_samples_split = [2,5,34,67,200]
        min_samples_leaf = [1,2,6,20]
        
        random_grid = {'n_estimators':n_estimators,
                    'max_features':max_feature,
                    'max_depth':max_depth,
                    'min_samples_split':min_samples_split,
                    'min_samples_leaf':min_samples_leaf}

        from sklearn.model_selection import RandomizedSearchCV

        re_regressor = RandomizedSearchCV(estimator=model,
                                        param_distributions=random_grid,
                                        scoring='neg_mean_squared_error',
                                        cv=5,
                                        verbose=2,
                                        random_state=42,
                                        n_jobs=1)
        re_regressor.fit(x_train,y_train)
        predicit = re_regressor.predict(x_train)
        print("The predict value is : ",re_regressor.predict([[n]]))

        # the mean_absolute_error values
        mae = mean_absolute_error(y_train,predicit)
        # the mean_squared_error values
        mse = mean_squared_error(y_train, predicit)
        print("the Mean absolute error : ",mae)
        print("the Mean squared error : ",mse)

    
    decision_tree(x, y, n)
    linear_regression(x, y, n)
    Random_forest_regression(x,y,n)


# pasting the dataset link or path link :: 
d = str(input("paste the link of dataset : "))
data = pd.read_csv(f"{d}")

# the feature columns::
feature = str(input(f"enter the feature column  : "))
## the target columns :: 
target = str(input("enter the target column : "))

## to predicting values::
n = int(input('Enter the value to predict: '))

x = np.array(data[feature])
y = np.array(data[target])
x = x.reshape(-1,1)

model(x,y,n)

