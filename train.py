import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from scipy import sparse
from utils import error_metrics

dataset_file = '../../Desktop/Case_Study_GA/dataset_product.pickle'

train_matrix_file = '../../Desktop/Case_Study_GA/X_train_product.npz'

val_matrix_file = '../../Desktop/Case_Study_GA/X_val_product.npz'

with open(dataset_file, 'rb') as handle:
    dataset = pickle.load(handle)

X_train_matrix = sparse.load_npz(train_matrix_file)
y_train = dataset['y_train']
y_train = np.delete(y_train, 72576)  # Element that was removed in cleaning


X_val_matrix = sparse.load_npz(val_matrix_file)
y_val = dataset['y_val']

dtype = torch.float
device = torch.device("cpu")

# Get data, initialize tensors
train_length = int(X_train_matrix.shape[0]/1)
inputs = X_train_matrix[:train_length]  # Matrix
targets = y_train[:train_length]  # Column/Series

val_inputs = X_val_matrix  # Matrix
val_targets = y_val  # Column/Series

print(inputs.shape, targets.shape)
print(val_inputs.shape, val_targets.shape)

x_test = torch.tensor(val_inputs.todense(), dtype=torch.float32)
y_test = torch.tensor(val_targets, dtype=torch.long)

print(x_test.shape, y_test.shape)

k = open('../../Desktop/Case_Study_GA/label_encoder_product.pickle', 'rb')
le = pickle.load(k)

# Parameters
N, D_in, D_out = inputs.shape[0], inputs.shape[1], len(le.classes_)
H = 128

# Model: Feed-forward neural network
model = torch.nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
)

# Hyper-parameters
learning_rate = 0.0005
batch_size = 149

# Regularization
#weight_decay=0.00005

# ADAM
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss
criterion = nn.CrossEntropyLoss()

# To collect training statistics
loss_hist, val_loss_hist = [], []

# Train
num_epochs = 11
epochs = range(num_epochs)
for t in epochs:
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    for batch in range(0, int(N / batch_size)):
        # Get batch

        x = inputs
        y = targets

        batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
        batch_y = y[batch * batch_size: (batch + 1) * batch_size]

        batch_x = torch.tensor(batch_x.todense(), dtype=torch.float32)
        batch_y = torch.tensor(batch_y, dtype=torch.long)

        # Calculate prediction (forward step)
        outputs = model(batch_x)

        # Calculate loss
        loss = criterion(outputs, batch_y)

        # Calculate gradients and update weights (backward step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Collect statistics every epoch
    loss_hist.append(loss.item())

    state = {'epoch': t,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss': loss,
             'loss_hist': loss_hist}  # Fill in rest here??

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    outputs_val = model(x_test)
    val_loss = criterion(outputs_val, y_test)
    val_loss_hist.append(val_loss.item())
    val_pred = torch.max(outputs_val.data, 1)[1]
    val_true = y_test

    # Print every .... epochs
    if t % 1 == 0:
        print('Epoch {}/{}'.format(t, num_epochs - 1))
        print('-' * 10)
        print('Train loss: ' + str(loss.item()))
        print('Validate loss: ' + str(val_loss.item()))
        print(error_metrics.misclassified(val_true, val_pred))

    state_file = 'trained_models/train_h128_product_STATE' + str(t) + '.tar'
    model_file = 'trained_models/train_h128_product_MODEL' + str(t) + '.pt'

    #torch.save(state, state_file)
    torch.save(model, model_file)

# Save statistics
loss_hist_df = pd.DataFrame(loss_hist)
loss_hist_df.columns = ['Train']
loss_hist_df['Validate'] = val_loss_hist
loss_hist_df.to_csv('train_errors/loss_hist_train_h128_product.csv')
torch.save(model, 'trained_models/train_h128_product_MODEL_END.pt')
