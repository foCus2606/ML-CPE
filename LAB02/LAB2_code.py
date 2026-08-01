import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"D:\ML\dog_adoption_master.csv")

df_num = df.select_dtypes(include=np.number)

print(df_num.head())
print(df_num.shape)

#Part1
x = df[['weight_kg']]
y = df[['age_years']]

x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model_simple = LinearRegression()
model_simple.fit(x_train_reg , y_train_reg)

y_pred_simple = model_simple.predict(x_test_reg)
print('R2 =',r2_score(y_test_reg , y_pred_simple))

plt.figure(figsize=(6,4))
plt.scatter(x_test_reg,y_test_reg,alpha=0.3)
plt.plot(x_test_reg , y_pred_simple, color='red')
plt.xlabel('Weight (kg)')
plt.ylabel('Age (years)')
plt.title('Simple Liner Regression')
plt.show()
x = df[['weight_kg',
        'days_in_shelter',
        'aggression_score',
        'energy_level',
        'expectation_score']]
y = df['age_years']
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
    x, y, test_size=0.2, random_state=42
)


model_multi = LinearRegression()
model_multi.fit(x_train_reg,y_train_reg)
y_pred_multi = model_multi.predict(x_test_reg)

print('R2 =',r2_score(y_test_reg, y_pred_multi))
print('Coefficients =', model_multi.coef_)

sample = [[15,10,2,4,8]]
pred_age = model_multi.predict(sample)

print('Predicted age =', pred_age[0], 'years')

mae = mean_absolute_error(y_test_reg, y_pred_multi)
mse = mean_squared_error(y_test_reg, y_pred_multi)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_multi)

print('MAE =', mae)
print('MSE =', mse)
print('RMSE =',rmse)
print('R2 =',r2)
#Part2
x = df[['age_years',
        'weight_kg',
        'days_in_shelter',
        'aggression_score',
        'energy_level',
        'expectation_score']]
y = df['returned']

x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(
    x,y,test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train_clf, y_train_clf)

y_pred = log_model.predict(x_test_clf)
print('Accuracy =', accuracy_score(y_test_clf, y_pred))
sample = [[2.0, 12, 5, 1, 3, 8]]
pred = log_model.predict(sample)
print('Predicted returned =', pred[0])# 0 = ไม่คืน, 1 = คืน
plt.figure(figsize=(6,5))
sns.scatterplot(
    data=df,
    x='aggression_score',
    y='expectation_score',
    hue='returned',
    alpha=0.4
)
plt.title('Returned vs Not Returned')
plt.show()
cm = confusion_matrix(y_test_clf, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(y_test_clf, y_pred))

#Part3
r2_simple = r2_score(y_test_reg, y_pred_simple)
r2_multi = r2_score(y_test_reg, y_pred_multi)

print('Simple R2 =', r2_simple)
print('Multiple R2 =', r2_multi)
train_score = model_multi.score(x_train_reg, y_train_reg)
test_score = model_multi.score(x_test_reg, y_test_reg)

print('Train score =', train_score)
print('Test score =', test_score)
print('Regression R2 =', r2_multi)
print('Classification Accuracy =', accuracy_score(y_test_clf, y_pred))
plt.figure(figsize=(12,10))
sns.heatmap(df_num.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
