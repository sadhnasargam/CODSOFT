import pandas as pd
import matplotlib.pyplot as plt
import joblib

df= pd.read_csv("advertising.csv")
print(df.head())
print(df.info())

#here we can find the staticqal summary
print(df.describe())
print(df.isnull().sum())

# here we see all the column of the dataset
print(df.columns)

# here we check that which features are realted to the sales and how much
print(df.corr(numeric_only=True))

# Relationship between TV advertising and Sales

plt.figure(figsize=(6,4))
plt.scatter(df["TV"], df["Sales"])

plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.show() 

# Select input features and target
X=df.drop("Sales",axis=1)
y=df["Sales"]
 
# Splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Made the model and train this
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#  evaluationg the metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Calculating Mean Absolute Error

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculating Mean Squared Error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculating R2 Score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
print("Model Accuracy (R2 Score):", round(r2*100,2), "%")

# Comparing actual and predicted sales
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.show()

# Predicting sales for new advertising values
new_data = pd.DataFrame([[230.1, 37.8, 69.2]],columns=["TV", "Radio", "Newspaper"])
predicted_sales = model.predict(new_data)
print("Predicted Sales:", predicted_sales[0])

# Creating a comparison table
comparison = pd.DataFrame({"Actual Sales": y_test,"Predicted Sales": y_pred})
print(comparison.head(10))

# Displaying the model coefficients
coefficients = pd.DataFrame({"Feature": X.columns,"Coefficient": model.coef_})
print(coefficients)

# Displaying the intercept
print("Intercept:", model.intercept_)

# Save the training model
joblib.dump(model, "sales_prediction_model.pkl")
print("Model saved successfully")

# Save actual and predicted sales into a CSV file
comparison.to_csv("Sales_prediction_result.csv",index=False)
print("Predicted result saved succssfully")
