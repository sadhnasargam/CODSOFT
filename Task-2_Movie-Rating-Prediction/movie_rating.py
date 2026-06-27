import pandas as pd
import joblib
import matplotlib.pyplot as plt
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')
print(df.head())
print(df.info())

# delete the missing values and duplicate values 
# Clean Year
df['Year'] = df['Year'].str.extract(r'(\d+)')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Clean Duration
df['Duration'] = df['Duration'].str.extract(r'(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Clean Votes
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
print(df.head())
print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
df['Actors'] = df['Actor 1'].astype(str) +" " + df['Actor 2'].astype(str) +" " + df['Actor 3'].astype(str)
df = df[['Year','Duration','Votes','Genre','Rating','Director','Actors']]

# convert strin data into the numeric form 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
categorical_features = ['Genre', 'Director', 'Actors']

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'),['Genre', 'Director', 'Actors'])
    ],
    remainder='passthrough'
)
print(df.head())

# Train-Test Split(here we inform the data who is input and who is output)

from sklearn.model_selection import train_test_split
X = df.drop('Rating',axis=1)
y = df['Rating']
X = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)


# here we train the model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("model is trained")

# model value prediction

y_pred = model.predict(X_test)

# model evaluatin means check the data accuracy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# data visualization

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.grid(True)

plt.show()

# Save the model and preprocessing 

joblib.dump(model, "movie_rating_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("Model and Preprocessor saved successfully")

