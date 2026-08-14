from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df1 = pd.read_csv("wine dataset/winequality-red.csv", sep=";")
df2 = pd.read_csv("wine dataset/winequality-white.csv", sep=";")
df1["type"] = 0
df2["type"] = 1

df = pd.concat([df1, df2], ignore_index=True)
#(0 para o vermelho) e (1 para o branco)

X = df.drop("type", axis=1)
y = df["type"]

X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.4,random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.5,random_state=42)
del X_,y_
#train 60%,cv 20%,test 20%

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42)

model.fit(X_train,y_train)

y_cv_pred = model.predict(X_cv)
print("CV accuracy", accuracy_score(y_cv, y_cv_pred))

y_pred = model.predict(X_test)
print("Test accuracy", accuracy_score(y_test, y_pred))

print("Confusion matrix", confusion_matrix(y_test, y_pred))
