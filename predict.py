import pickle
from nltk.corpus import stopwords
from utils import text_processing
import torch
from sklearn.metrics import accuracy_score

# Parameters
dataset_file = '../../../dataset_product.pickle'
stop = stopwords.words('english')
tf_file = '../../../tfidf_vec_product.pkl'
model_file_name = 'trained_models/train_h128_product_MODEL3.pt'
label_encoder_file_name = '../../../label_encoder_product.pickle'
save_results_csv_file_name = 'data/model_3_test_predictions.csv'

# Get Data
with open(dataset_file, 'rb') as handle:
    dataset = pickle.load(handle)
X_test_text = dataset['X_test']

y_test_bool = True
if y_test_bool == True:
    y_test = dataset['y_test']
    # Get true labels
    k = open(label_encoder_file_name, 'rb')
    le = pickle.load(k)
    true_labels = le.inverse_transform(y_test)
    k.close()


def predict(text, model_file_name, label_encoder_file_name):
    # Transform input text
    X_test_matrix = text_processing.clean_raw_text(text_as_series=text,
                                                   tf_file=tf_file,
                                                   stopwords=stop)
    # Prepare input data
    dtype = torch.float
    device = torch.device('cpu')
    x_test = torch.tensor(X_test_matrix.todense(), device=device, dtype=dtype)

    # Load Model
    model = torch.load(model_file_name)

    # Predict
    outputs = model(x_test)
    y_pred = torch.max(outputs.data, 1)[1]
    predicted = y_pred.numpy()

    # Get label encoder to transform prediction output
    k = open(label_encoder_file_name, 'rb')
    le = pickle.load(k)


    # Return prediction for sample
    predicted_label = le.inverse_transform(predicted)
    k.close()
    return predicted_label

if __name__ == "__main__":
    predictions = predict(text=X_test_text,
                          model_file_name=model_file_name,
                          label_encoder_file_name=label_encoder_file_name)

    if y_test_bool == True:
        # Quick evaluation metrics
        print('Accuracy: ' + str(round(accuracy_score(true_labels, predictions), 4)))
        # Save results
        results_df = pd.DataFrame({'True': true_labels, 'Pred': predictions})
        results_df.to_csv(save_results_csv_file_name, index=False)
        print('End')
    else:
        print(predictions)

