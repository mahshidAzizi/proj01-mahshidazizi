import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# خواندن داده از فایل CSV
data = pd.read_csv('Hotel Reservations.csv')

label_encoder = LabelEncoder()

# لیست نام ستون‌هایی که می‌خواهید تبدیل شوند
categorical_columns = ['Booking_ID', 'type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']

# تبدیل داده‌های رشته‌ای به عدد
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
features = data.iloc[:, 1:].values

# استانداردسازی داده‌ها
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# استانداردسازی داده‌ها
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# تعداد خوشه‌ها از 2 تا 10
num_clusters_range = range(2, 11)

# آماده‌سازی لیست برای ذخیره SSE هر خوشه
#sse_list = []

# انجام خوشه‌بندی و محاسبه SSE برای هر تعداد خوشه
#for num_clusters in num_clusters_range:
 #   kmeans = KMeans(n_clusters=num_clusters, random_state=42)
 #   kmeans.fit(data_scaled)
 #   sse_list.append(kmeans.inertia_)

# نمایش شکل خوشه‌بندی
  #  plt.scatter(features_scaled[:, 0], features_scaled[:, 1],features_scaled[:, 2], c=kmeans.labels_, cmap='viridis', alpha=0.5)
   # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200)
    #plt.title(f'K-means Clustering (Num Clusters: {num_clusters})')
    #plt.xlabel('Feature 1')
    #plt.ylabel('Feature 2')
    #plt.show()
    
# رسم نمودار SSE
#plt.plot(num_clusters_range, sse_list, marker='o')
#plt.xlabel('تعداد خوشه‌ها')
#plt.ylabel('SSE (Sum of Squared Errors)')
#plt.title('نمودار SSE بر حسب تعداد خوشه‌ها')
#plt.show()

# آماده‌سازی لیست برای ذخیره متغیرهای مختلف هر خوشه
cluster_labels_list = []

# انجام خوشه‌بندی برای هر تعداد خوشه
for num_clusters in num_clusters_range:
    dbscan = DBSCAN(eps=1, min_samples=num_clusters)
    cluster_labels = dbscan.fit_predict(features_scaled)
    cluster_labels_list.append(cluster_labels)

    # نمایش شکل خوشه‌بندی
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title(f'DBSCAN Clustering (Min Samples: {num_clusters})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# نمایش لیست خوشه‌ها برای هر تعداد خوشه
for i, num_clusters in enumerate(num_clusters_range):
    print(f'Cluster Labels for DBSCAN (Min Samples: {num_clusters}): {cluster_labels_list[i]}')
