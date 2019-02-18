from flask import Flask, render_template, url_for, request
from nltk.corpus import stopwords
import torch
from utils import text_processing
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Trained model
    model = torch.load('../trained_models/train_h128_product_MODEL3.pt')

    if request.method == 'POST':
        message = request.form['message']
        data = pd.Series(message)

        # Pre-processing Text
        # Load TfidfTransformer()
        X_test_matrix = text_processing.clean_raw_text(text_as_series=data,
                                                       tf_file='../../../Desktop/Case_Study_GA/tfidf_vec_product.pkl',
                                                       stopwords=stopwords.words('english'))
        # Prepare input data
        dtype = torch.float
        device = torch.device('cpu')
        x_test = torch.tensor(X_test_matrix.todense(), device=device, dtype=dtype)
        outputs = model(x_test)
        y_pred = torch.max(outputs.data, 1)[1]
        my_prediction = y_pred.numpy()

        # Load label encoder
        k = open('../../../Desktop/Case_Study_GA/label_encoder_product.pickle', 'rb')
        le = pickle.load(k)
        pred_label = le.inverse_transform(my_prediction)
        print(my_prediction, pred_label)

    return render_template('result.html', prediction=pred_label)


if __name__ == '__main__':
    app.run(debug=True)
