 # Step 0: Import The Libraries 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import r2_score



# Step 1: import and Clean the Data 
try:
    #Replace with path of your file
    file = pd.read_csv("cars_data_clean.csv")
except FileNotFoundError:
    print("File Not Found Error!")

file = file.drop(columns=["usedCarSkuId","ip", "images", "imgCount", "threesixty",
                        "dvn", "oem", "top_features",
                        "comfort_features", "interior_features", "exterior_features",
                        "safety_features", "Color", "exterior_color", "model_type_new"], axis=1)
#converting object columns to integers
object_columns_to_convert = ["loc", 'transmission', "body", "Valve Configuration",
                             "Front Brake Type", "Rear Brake Type",
                             "fuel", "model", "variant", "City",
                             "Tyre Type", "Steering Type", 
                             "Turbo Charger", "Super Charger",
                             "utype", "carType", "Engine Type",
                             "Gear Box", "Drive Type", "Turning Radius",
                             "owner_type", "Fuel Suppy System", "owner_type",
                              'Seats',"state", 'Doors']
#str to number
for col in object_columns_to_convert:
    file[col] = pd.to_numeric(file[col], errors='coerce').fillna(0).astype(int)


#Nan to number
for col in (list(file.columns)):
    file[col] = file[col].fillna(0).astype(int)
    
print(file.dtypes)


# Step 2: Split the Data(train, Test) 
y = file[["listed_price"]]
X = file[['myear', 'km', 'No of Cylinder', 'Valves per Cylinder', 
                   'Length', 'Width', 'Height', 'Wheel Base', 'Kerb Weight', 
                   'Gross Weight', 'Seats', 'Top Speed', 'Acceleration', 
                   'Doors', 'Cargo Volume', 'Compression Ratio', 
                   'Alloy Wheel Size', 'Ground Clearance Unladen', 
                   'Max Power Delivered', 'Max Power At', 
                   'Max Torque Delivered', 'Max Torque At', 'Bore', 'Stroke']]
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=8)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# Step 3: Create model 
#from sklearn import linear_modelpoly
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(X_train)
clf = linear_model.LinearRegression()



# Step 4: Train the Model 
train_y_ = clf.fit(train_x_poly, y_train)

# Step 5: Make Predictions 
test_x_poly = poly.fit_transform(X_test)
test_y_ = clf.predict(test_x_poly)

# Step 6: Evaluate 
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test,test_y_ ) )