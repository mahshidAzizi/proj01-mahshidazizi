import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df=pd.read_csv('Hotel Reservations.csv')
# انتخاب ویژگی‌های عددی
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
# ایجاد یک MinMaxScaler
scaler = MinMaxScaler()
# نرمال‌سازی ویژگی‌های عددی
df[numeric_features] = scaler.fit_transform(df[numeric_features])
# انتخاب ویژگی‌های غیر عددی
non_numeric_features = df.select_dtypes(exclude=['float64', 'int64']).columns
# ایجاد یک LabelEncoder
label_encoder = LabelEncoder()
# تبدیل ویژگی‌های غیر عددی به اعداد
df[non_numeric_features] = df[non_numeric_features].apply(lambda col: label_encoder.fit_transform(col))
X = df.drop('booking_status', axis=1)
y = df['booking_status']

# تقسیم داده‌ها به داده‌های آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# gini
tree_model = DecisionTreeClassifier(criterion='gini', random_state=42)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
df['booking_status'] = df['booking_status'].astype(str)

#entropy
tree_model_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

# آموزش مدل
tree_model_entropy.fit(X_train, y_train)

# پیش‌بینی برچسب‌ها برای داده‌های آزمون
y_pred_entropy = tree_model_entropy.predict(X_test)

# محاسبه دقت
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
print(f'Accuracy (Entropy): {accuracy_entropy}')

# نمایش درخت به صورت گرافیکی
plt.figure(figsize=(20,10))
plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=df['booking_status'].unique())
plt.show()


