import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# خواندن داده از فایل CSV
data = pd.read_csv('Hotel Reservations.csv')

label_encoder = LabelEncoder()


# لیست نام ستون‌هایی که می‌خواهید تبدیل شوند
categorical_columns = ['Booking_ID', 'type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']

# تبدیل داده‌های رشته‌ای به عدد
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
features = data.iloc[:, 1:].values

# جدا کردن ویژگی‌ها و برچسب‌ها
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# تقسیم داده به دو قسمت آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_algorithm(algorithm, X_train, X_test, y_train, y_test):
    # آموزش الگوریتم
    algorithm.fit(X_train, y_train)

    # پیش‌بینی
    y_pred = algorithm.predict(X_test)

    # محاسبه ماتریس اشتباهات
    cm = confusion_matrix(y_test, y_pred)

    # محاسبه معیارهای ارزیابی
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'Confusion Matrix': cm,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def plot_roc_curve(algorithm, X_test, y_test, name):
    # رسم نمودار ROC
    plt.figure(figsize=(10, 6))
    # محاسبه احتمالات
    y_proba = algorithm.predict_proba(X_test)[:, 1]
    # محاسبه نقاط در نمودار ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    # رسم نمودار
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show()

# فرض شده: داده‌های شما در متغیرهای X و y قرار دارند
# X: ویژگی‌ها
# y: برچسب‌ها

# تقسیم داده به دو قسمت آموزش و آزمون
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# لیستی برای ذخیره نتایج
#results = []

# تعریف الگوریتم‌ها
#algorithms = {
#    'Decision Tree': DecisionTreeClassifier(random_state=42),
#    'SVM': SVC(random_state=42),
#    # ... سایر الگوریتم‌ها
#}

# آموزش و ارزیابی هر الگوریتم
#for name, algorithm in algorithms.items():
    # ارزیابی الگوریتم و ذخیره نتایج
 #   results.append({
  #      'Algorithm': name,
   #     **evaluate_algorithm(algorithm, X_train, X_test, y_train, y_test)
    #})

    # رسم نمودار ROC
    #plot_roc_curve(algorithm, X_test, y_test, name)

# ایجاد DataFrame از نتایج
#results_df = pd.DataFrame(results)

# نمایش DataFrame
#print(results_df)
