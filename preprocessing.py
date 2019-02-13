import pandas as pd
from sklearn.model_selection import train_test_split
from features import text_processing
import pickle
from scipy import sparse
import time


# Needs to be better organized from here
data = text_processing.get_nonempty_df(file='../../Desktop/Consumer_Complaints/Consumer_Complaints.csv',
                                       column_dict={'Product': 'Product', 'Consumer complaint narrative': 'Text'})

data = text_processing.consolidate_features(df=data,
                                            target_col_name='Product',
                                            feature_dict={'Other loans': [0, 3, 4, 6],
                                                          'Other accounts and services': [1, 2, 5, 7, 8, 9, 10]})

data, classes = text_processing.encode_labels(df=data, target_col_name='Product')

data = text_processing.remove_redacted(df=data, col_name='Text')

# To here

# Split into test and train before applying CountVectorizer()
X_data = data.Text
y_data = data.Product
print(y_data.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=17)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Export test_data
start = time.time()
print('Exporting test data')
test_df = pd.DataFrame(X_test)
test_df['Label'] = y_test
print(test_df.head())
test_df.to_csv('data/test_data.csv', index=False)
end = time.time()
print('Finished exporting test data in ' + str(round(end-start, 3)) + 's')

print(X_train.shape, type(X_train)) # ==> (167386,), pandas Series

# Apply TfidfVectorizer() and save the fitted vectorizer
X_train = text_processing.transform_text(X_train)

print(X_train.shape, type(X_train)) # ==> (167386, 75610)


# Save items to be applied/used later
start = time.time()
print('Saving labels')
pickle.dump(classes, open("classes.pickle", "wb"))
end = time.time()
print('Finished saving in ' + str(round(end-start, 3)) + 's')

print(X_train.toarray().shape)  # ==> (167386, 75610)
print(type(X_train))  # ==> scipy.sparse.csr.csr_matrix
print(y_train.value_counts())

# Export train_data
start = time.time()
print('Exporting train data')
sparse.save_npz('data/X_train_csr_mat.npz', X_train)
y_train.to_csv('data/y_train.csv', index=False)
end = time.time()
print('Finished exporting train data in ' + str(round(end-start, 3)) + 's')
print('End')
