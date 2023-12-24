import matplotlib # %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# %%
data = pd.read_csv('D:\Daneshga\Hoosh\P\creditcard.csv')
data.head()

# %%
data.info()

# %%
kmeans = KMeans(n_clusters=2, n_init='auto')
x = data.drop("Class", axis=1)
kmeans.fit(x)
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_
plt.figure(figsize=(8, 6))
plt.scatter(x['Time'], x['V2'], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
plt.title('K-means Clustering')
plt.show()

# %%
fraud_count = [0,0]
for i in range (data.shape[0]):
    if data['Class'][i] == 1:
        fraud_count[cluster_labels[i]] +=1

fraud_count

# %%
count = [0,0]
for i in range (len(cluster_labels)):
    count[cluster_labels[i]] += 1

count

# %%
x = fraud_count[0] / count[0] * 100
y = fraud_count[1] / count[1] * 100
print('Percentage of fraud in cluster 1:', x)
print('Percentage of fraud in cluster 2:', y)

# %%

sns.histplot(data['Class'])
plt.yscale('log')
plt.show()

# %%
x = data.drop(columns='Class', axis=1)
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123)

# %%
oversampler = SMOTE()
x_train, y_train = oversampler.fit_resample(x_train, y_train)

# %%

sns.histplot(y_train)
plt.show()

# %%
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(f'x_train{x_train.shape}\n, x_test{x_test.shape}\n, y_train{y_train.shape}\n, y_test{y_test.shape}')

# %%
def svm_model(x_train, y_train, x_test, y_test):
    svm = SVC()
    svm.fit(x_train, y_train)
    y_train_pred = svm.predict(x_train)
    y_train_cl_report = classification_report(y_train, y_train_pred, target_names=['No Fraud', 'Fraud'])
    print("_" * 100)
    print("TRAIN MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_train_cl_report)
    y_test_pred = svm.predict(x_test)
    y_test_cl_report = classification_report(y_test, y_test_pred, target_names=['No Fraud', 'Fraud'])
    print("_" * 100)
    print("TEST MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_test_cl_report)
    print("_" * 100)
    return y_test_pred, svm

# %%
y_test_pred, svm = svm_model(x_train, y_train, x_test, y_test)

# %%
def logistic_regression(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(penalty='l2', C=1.0, random_state=42, solver='liblinear')
    lr.fit(x_train, y_train)
    y_train_pred = lr.predict(x_train)
    y_train_cl_report = classification_report(y_train, y_train_pred, target_names=['No Fraud', 'Fraud'])
    print("_" * 100)
    print("TRAIN MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_train_cl_report)
    y_test_pred = lr.predict(x_test)
    y_test_cl_report = classification_report(y_test, y_test_pred, target_names=['No Fraud', 'Fraud'])
    print("_" * 100)
    print("TEST MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_test_cl_report)
    print("_" * 100)
    return y_test_pred, lr

# %%
y_test_pred, lr = logistic_regression(x_train, y_train, x_test, y_test)


