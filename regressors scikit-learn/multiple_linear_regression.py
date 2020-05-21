# Multiple Linear Regression

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# Importing the dataset
dataset = pd.read_csv('csv_teste_regressor.csv')
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Un')))]

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.model_selection import train_test_split
from yellowbrick.regressor import PredictionError

# Instantiate the linear model and visualizer
from sklearn.linear_model import LinearRegression
model = LinearRegression()
visualizer = PredictionError(model)
regressor=LinearRegression()
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)


visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure

from sklearn.metrics import explained_variance_score
print ('explained_variance_score:')
print (explained_variance_score(y_test, y_pred))

from sklearn.metrics import max_error
print ('max_error:')
print (max_error(y_test, y_pred))

from sklearn.metrics import mean_absolute_error
print ('mean_absolute_error:')
print (mean_absolute_error(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print ('mean_squared_error:')
print (mean_squared_error(y_test, y_pred))

# sum(n < 0 for n in y_test)
# sum(n < 0 for n in y_pred)

from sklearn.metrics import mean_squared_log_error
print ('mean_squared_log_error:')
print (mean_squared_log_error(y_test, y_pred))

from sklearn.metrics import median_absolute_error
print ('median_absolute_error:')
print (median_absolute_error(y_test, y_pred))

from sklearn.metrics import r2_score
print ('r2_score:')
print (r2_score(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
n = len(y_pred) #size of test set
p = len(X_test[0]) #number of features
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print ("adjusted_r2:")
print (adjusted_r2)
#have to check
#from sklearn.metrics import mean_tweedie_deviance
#ESTA PREVENDO NUMEROS NEGATIVOS EMBORA EU SO TENHA NUMEROS POSITIVOS
