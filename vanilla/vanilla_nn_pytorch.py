import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split dataset into train and test
train_dataset, test_dataset = numpy.split(dataset, [round(dataset.shape[0]*0.80)])

# split into input (X) and output (Y) variables
X_train = torch.from_numpy(train_dataset[:,0:8])[:600]
Y_train = torch.from_numpy(train_dataset[:,8])[:600]

X_test = torch.from_numpy(test_dataset[:,0:8])
Y_test = torch.from_numpy(test_dataset[:,8])


# Hyper Parameters
input_size = 8
num_classes = 1
hidden_size = 12
num_epochs = 150
batch_size = 32
learning_rate = 0.001

train = data_utils.TensorDataset(X_train, Y_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)

test = data_utils.TensorDataset(X_test, Y_test)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)

# Neural Network Model (2 hidden layers)
class VanillaNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(VanillaNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, input_size)
        self.hidden_layer2 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


model = VanillaNet(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        features = Variable(features).float()
        labels = Variable(labels.view(-1, 1)).float()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # if (i+1) % 10 == 0:
        #     print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
        #            %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        print ('Epoch [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, loss.data[0]))


# Test the Model
correct = 0
total = 0
for features, labels in test_loader:
    features = Variable(features).float()
    outputs = F.sigmoid(model(features))
    predicted = torch.round(outputs.data).float().squeeze()
    total += labels.size(0)
    correct += (predicted == labels.float()).sum()


print('\nacc: %.2f%%' % ((correct / total) * 100))