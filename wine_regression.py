from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df1 = pd.read_csv("data/winequality-red.csv", sep=";")
df2 = pd.read_csv("data/winequality-white.csv", sep=";")

df = pd.concat([df1, df2], ignore_index=True)

X = df.drop("quality", axis=1)
y = df["quality"]

#sns.pairplot(df)
#print(plt.show())

X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.4,random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.5,random_state=42)
del X_,y_
#train 60%,cv 20%,test 20%

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Test mean absolute error", mean_absolute_error(y_test, y_pred))
print("Test root mean squared error", mean_squared_error(y_test, y_pred) ** 0.5)
print("Test R2 score", r2_score(y_test, y_pred))

y_cv_pred = model.predict(X_cv)

print("CV mean absolute error", mean_absolute_error(y_cv, y_cv_pred))
print("CV root mean squared error", mean_squared_error(y_cv, y_cv_pred) ** 0.5)
print("CV R2 score", r2_score(y_cv, y_cv_pred))