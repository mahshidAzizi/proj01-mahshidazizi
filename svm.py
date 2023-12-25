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
from sklearn import svm
from sklearn.datasets import make_classification


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
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# تقسیم داده‌ها به دو بخش آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ایجاد مدل SVM با هسته خطی
model_linear = svm.SVC(kernel='linear')
model_linear.fit(X_train, y_train)

# نمایش داده‌ها در شکل
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='.')

# نمایش داده‌های آزمون
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='x', label='Test Data')

# رسم مرز تصمیم‌گیری
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

 ایجاد یک مشکل (meshgrid) از نقاط بین داده‌ها
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = model_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])

# نمایش مرز تصمیم‌گیری
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

plt.title('Linear SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
y_pred_linear = model_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Linear Kernel Accuracy:", accuracy_linear)
 ایجاد مدل SVM با هسته چندجمله‌ای
model_poly = svm.SVC(kernel='poly', degree=3)
model_poly.fit(X_train, y_train)
y_pred_poly = model_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print("Polynomial Kernel Accuracy:", accuracy_poly)

# نمایش داده‌ها در شکل
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='.')

# نمایش داده‌های آزمون
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='x', label='Test Data')

# رسم مرز تصمیم‌گیری
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

 ایجاد یک مشکل (meshgrid) از نقاط بین داده‌ها
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = model_poly.decision_function(np.c_[xx.ravel(), yy.ravel()])

# نمایش مرز تصمیم‌گیری
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

plt.title('Polynomial SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
# ایجاد مدل SVM با هسته گاوسی (RBF)
model_rbf = svm.SVC(kernel='rbf')
model_rbf.fit(X_train, y_train)
y_pred_rbf = model_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("RBF Kernel Accuracy:", accuracy_rbf)


# نمایش داده‌ها در شکل
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='.')

# نمایش داده‌های آزمون
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='x', label='Test Data')

# رسم مرز تصمیم‌گیری
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# ایجاد یک مشکل (meshgrid) از نقاط بین داده‌ها
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = model_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# نمایش مرز تصمیم‌گیری
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

plt.title('RBF SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()