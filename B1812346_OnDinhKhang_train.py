import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

dataFrame = pd.read_csv('./iris_dataset.csv')
df = dataFrame.iloc[:, 0:4]
y = dataFrame.variety
# print(df.describe())
# print("Kiem tra xem du lieu co bi thieu (NULL) khong?")
# print(df.isnull().sum())

# train test split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=100)

# Xây dựng mô hình với k = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Tính độ chính xác
print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
      model.score(X_test, y_test))
# Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: 0.978

# Save model
file = open('./model_knn', 'wb')
pickle.dump(model, file)

# # load the model from disk
# test = [[7, 3.2, 4.7, 1.4]]
# loaded_model = pickle.load(open('model_knn', 'rb'))
# result = loaded_model.predict(test)
# print(result)
