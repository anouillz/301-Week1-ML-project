import pandas as pd
import numpy as np
from isc301.config import housing_prices_raw_path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


#def data_preprocessing(df: pd.DataFrame) -> (np.array, np.array):
def data_preprocessing(df: pd.DataFrame):   

    def remove_outliers(dframe, col):
        # get outliers 
        q1 = dframe[col].quantile(0.25)
        q3 = dframe[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # remove outliers 
        dframe = dframe[(dframe[col] >= lower_bound) & 
                    (dframe[col] <= upper_bound)]
        
        return dframe

    
    # cleaning data 
    data = df.copy()
    data = data.dropna()
    data = data.drop_duplicates()

    # removing outliers 
    data = remove_outliers(data, 'surf_hab')
    data = remove_outliers(data, 'surface_sous_sol')
    
    # see if a house is luxe or not (both conditions)
    data['is_luxe'] = np.where((data['qualite_materiau'] > 8) & (data['n_garage_voitures'] > 3) & (data['surf_hab'] > 8000), 1, 0)



    # from data make the np arrays
    Y = data['prix'].to_numpy(dtype=float)
    X = data[['surf_hab', 'qualite_materiau', 'n_garage_voitures', 'surface_sous_sol', 'n_pieces', 'is_luxe']].to_numpy(dtype=float)

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=60)

    # split data into validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=60) # 0.25 x 0.8 = 0.2

    # standardize the data
    scaler_standard = StandardScaler()
    X_train = scaler_standard.fit_transform(X_train)
    X_test = scaler_standard.transform(X_test)
    X_val = scaler_standard.transform(X_val)

    return X_train, y_train, X_test, y_test, X_val, y_val, scaler_standard

# Previous model used for overfitting demonstration
def model_fit_overfitting(X: np.array, Y: np.array):
    poly = PolynomialFeatures(degree=7)
    X_poly = poly.fit_transform(X)
    regr = LinearRegression()
    regr.fit(X_poly, Y)
    return regr, poly

# Final model used, with Lasso regularization to prevent overfitting
def model_fit(X:np.array,Y:np.array):
    p = 3  
    poly = PolynomialFeatures(degree=p)
    X_poly = poly.fit_transform(X)
    model = Lasso(200, max_iter=10000)

    model.fit(X_poly, Y)
    return model, poly

# Make predictions
def model_predict(model, poly, X: np.array):
    X_poly = poly.transform(X)
    return model.predict(X_poly)


#### Example on how to use functions

df = pd.read_csv(housing_prices_raw_path)

## get cleaned data: 
X, Y, X_test, Y_test, X_val, Y_val, scaler = data_preprocessing(df)

## our models
# final model 
model, poly = model_fit(X, Y)

# overfitting demonstration
modelO, polyO = model_fit_overfitting(X, Y)

## predictions
# predict model
y_pred = model_predict(model, poly, X)
y_pred_val = model_predict(model, poly, X_val)

# predict overfitting model
y_predO = model_predict(modelO, polyO, X)
y_predO_val = model_predict(modelO, polyO, X_val)
