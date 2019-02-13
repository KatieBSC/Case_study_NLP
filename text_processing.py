import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle


# Get data, remove NaN samples
def get_nonempty_df(file, column_dict):
    df = pd.read_csv(file)
    df = df[list(column_dict.keys())].copy().dropna(how='any')
    return df.rename(columns=column_dict)


# Consolidate features
def consolidate_features(df, target_col_name, feature_dict):
    counts = df[target_col_name].value_counts()
    less_common_products = [counts.index.tolist()[i] for i in range(len(counts)) if counts[i] < counts.mean()]
    for key in feature_dict.keys():
        lst = [less_common_products[idx] for idx in feature_dict[key]]
        df[target_col_name] = df[target_col_name].replace(lst, key)
    return df


# Encode the targets
def encode_labels(df, target_col_name):
    le = LabelEncoder()
    df[target_col_name] = le.fit_transform(df[target_col_name])
    return df, le.classes_


# Remove the XXs and numbers
def remove_redacted(df, col_name):
    df[col_name] = df[col_name].str.replace('X', '')
    df[col_name] = df[col_name].replace(to_replace='numeric')
    return df


# Convert text to binary and report on time
def transform_text(df):
    print('Start vectorizing')
    start = time.time()
    vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english').fit(df)
    pickle.dump(vectorizer, open('cv.pkl', 'wb'))
    vec_train = vectorizer.transform(df)
    end = time.time()
    print('Finished in ' + str(round(end-start, 3)) + 's')
    return vec_train
