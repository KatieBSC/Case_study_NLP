import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.model_selection import train_test_split


dtype = torch.float
device = torch.device("cpu")

# Get data
X_train = sparse.load_npz('data/X_train_csr_mat.npz')
y_train = pd.read_csv('data/y_train.csv', header=None)

print(X_train.toarray().shape)  # ==> (167386, 75610)
print(y_train.shape)  # ==> (167386, 1)
print()

X_train, X_rest, y_train, y_rest = train_test_split(X_train, y_train, test_size=0.66, random_state=17)

inputs = X_train
targets = y_train.values

print(type(inputs))  # ==> scipy.sparse.csr.csr_matrix
print(type(targets))  # ==> numpy.ndarray
print(inputs.shape)  # ==> (56911, 75610)
print(targets.shape)  # ==> (56911, 1)

N = inputs.shape[0]  # ==> 56911
D_in = inputs.shape[1]  # ==> 75610
D_out = targets.max() + 1  # ==> 9
H = 128

print(N, D_in, D_out, H)

inputs = inputs.toarray()

print(inputs.shape)  # ==> (56911, 75610)

x = torch.tensor(inputs, device=device, dtype=dtype)
y = torch.tensor(targets, device=device, dtype=torch.long).squeeze()

# Hyper-parameters
learning_rate = 0.005
batch_size = 64

# Neural Network with one hidden layer
model = torch.nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
loss_hist = []

# Train
epochs = range(10)
# idx = 0
for t in epochs:
    for batch in range(0, int(N / batch_size)):

        # Calculate batch
        batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
        batch_y = y[batch * batch_size: (batch + 1) * batch_size]

        # Forward step
        outputs = model(batch_x)

        # Calculate errors
        loss = criterion(outputs, batch_y)

        # Backward step (gradients and weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate the errors (Print errors every ... iterations)
    if t % 1 == 0:
        loss_hist.append(loss.item())
        print(t, loss.item())

# Save and export trained model and training errors
torch.save(model, 'trained_models/onethird_h128.pt')
loss_hist_df = pd.DataFrame(loss_hist)
loss_hist_df.to_csv('train_errors/loss_hist_h128_onethird.csv')
