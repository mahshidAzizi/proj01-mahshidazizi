import pandas as pd  
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv('Hotel Reservations.csv')  
print(data) 
data.head()  
column_name = 'repeated_guest'  # Replace with the actual column name
number_of_rows = data.shape[0]
missing_data_count = data.isnull().sum()
max_value_column = data[column_name].max()
min_value_column = data[column_name].min()
median_value_column = data[column_name].median()


print("Number of data points:", number_of_rows)
print("\nNumber of missing values in each column:")
print(missing_data_count)
print(f"Maximum value in {column_name}: {max_value_column}")
print(f"Minimum value in {column_name}: {min_value_column}")
print(f"Median value in {column_name}: {median_value_column}")
plt.figure(figsize=(10,40))
for i,col in enumerate(data.drop(['Booking_ID','lead_time','arrival_date','avg_price_per_room','no_of_previous_bookings_not_canceled'],axis=1).columns):
    ax = plt.subplot(14,2,i+1)
    sns.countplot(y=data[col])
    plt.title(col)
    plt.ylabel(None)
    plt.tight_layout()
    plt.show()

# Select only numeric columns
numeric_df = data.select_dtypes(include='number')
# Calculate correlation
correlation_matrix = numeric_df.corr()

#heatmap
# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(15,15),layout='constrained')
plt.title('Corelation Matrix')
sns.heatmap(correlation_matrix,annot=True)
plt.show()
# Checling important features according to target variable
cor=data.corr()
target=cor['booking_status_encoder'].drop('booking_status_encoder')
target_s=target.sort_values(ascending=False)
plt.figure(figsize=(10,10),layout='constrained')
plt.title('Important Features According To Target Variable')
sns.heatmap(target_s.to_frame(),annot=True)
plt.show()


nominal_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type','booking_status']
# Subset the DataFrame to include only nominal columns
nominal_df = data[nominal_columns]
# Get the number of occurrences for each unique value in each nominal column
nominal_counts = nominal_df.apply(lambda x: x.value_counts())
# Print the results
print("Number of occurrences for each unique value in each nominal column:")
print(nominal_counts)