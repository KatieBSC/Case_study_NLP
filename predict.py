import torch
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

# Get test data
test_data = pd.read_csv('data/test_data.csv')
X_test = test_data.Text
y_test = test_data.Label
print(X_test.shape)  # ==> (82444,)
print(y_test.shape)  # ==> (82444,)

# Load fitted transformer
tf1 = pickle.load(open("cv.pkl", 'rb'))

# Apply to test data
X_tf1 = tf1.transform(X_test)
print(X_tf1.toarray().shape)  # ==> (82444, 75610)
print(X_tf1.count_nonzero())


dtype = torch.float
device = torch.device('cpu')

# Choose some test data to predict
x_test = torch.tensor(X_tf1.toarray()[:20000], device=device, dtype=dtype)
y_test = torch.tensor(y_test[:20000], device=device, dtype=torch.long).squeeze()
print(x_test.shape)  # ==> 20000, 75610
print(y_test.shape)  # ==> 20000

# Load model
model = torch.load('trained_models/onethird_h128.pt')

# Predict
outputs = model(x_test)
y_pred = torch.max(outputs.data, 1)[1]
predicted = y_pred.numpy()
true = y_test.numpy()

# Example of output and labels for reference
print(predicted[:15])
print(true[:15])
k = open('classes.pickle', 'rb')
classes = pickle.load(k)
print(classes)
k.close()

# Quick evaluation metrics
print(accuracy_score(true, predicted))
print(confusion_matrix(true, predicted))

# Export results
