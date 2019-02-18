import re
import pickle


def clean_raw_text(text_as_series, tf_file, stopwords):
    tf = pickle.load(open(tf_file, 'rb'))
    series = text_as_series.reset_index(drop=True)

    def clean_num_punct(text):
        text = re.sub(r'([^a-zA-Z ]+?)', ' ', text)
        text = text.replace('X', '')
        text = text.replace('\n', ' ')
        return text.lower()

    series = series.apply(clean_num_punct)
    series = series.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords and len(word) > 1]))
    matrix = tf.transform(series)
    return matrix
