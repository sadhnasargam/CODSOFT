import pandas as pd
import joblib

# DATASET LOAD

df=pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print("\nNumber of rows and columns")
print(df.shape)
print("\nColumn name")
print(df.columns)
print("\nMissing values")
print(df.isnull().sum())         

#HANDLE THE MISSING VALUES

df["Age"].fillna(df["Age"].median(), inplace = True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace = True)
df.drop("Cabin", axis =1, inplace = True)
print("missing values after cleaning")
print(df.isnull().sum())

# WE WILL CONVERT TEXT DATA INTO NUMBER FORM

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# convert Gender column

df["Sex"] = le.fit_transform(df["Sex"])

# convert Embarked column

df["Embarked"] = le.fit_transform(df["Embarked"])
print("Data Converted Successfully")
print(df[["Sex", "Embarked"]].head())

# HERE WE WRITE THE INPUT AND OUTPUT DATA 

x = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
y = df["Survived"]
print(x.head())

# TRAINING & TESTING DATAST SPLIT

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("Ready Training Data")
print("Ready Testing Data")

# IMPORT RANDOM FOREST MODEL TRAINING

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
print("training complete")
prediction = model.predict(x_test)
print("Prediction complete")

# ACCURACY CHECK

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
print("model accuracy =", accuracy)

# CHECK THE MODEL PREDICTION 

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, prediction)
print("confusion_matrix")
print(cm)

# Make a new file

joblib.dump (model, "Titanik_Model.Pkl")
print("model saved")

