import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# Fetching Google's Stock prices from Quandl
df = quandl.get('WIKI/GOOGL')

# Selecting relevant columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Adding new columns based on values of existing columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Selecting columns from dataframe that we perceive to be features.
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

# Take a percentage of the length of dataset as number
# of days to shift label by.
forecast_out = int(math.ceil(0.01 * len(df)))

# Shift the value of label by the number of days
# determined by forecast_out.
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# Selecting features.
X = np.array(df.drop(['label'], 1))

# Standardize our data center to the mean.
X = preprocessing.scale(X)
df.dropna(inplace=True)

# Selecting label
y = np.array(df['label'])

# This "shuffles" our data and returns 2 pairs
# of data sets whose members are connected.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Selecting our Classifier
clf = LinearRegression()

# Train our Classifier using the training data
clf.fit(X_train, y_train)

# Testing our Classifier's accuracy.
# Comparing our actual data with what is predicted by our Classifier
accuracy = clf.score(X_test, y_test)

print accuracy


