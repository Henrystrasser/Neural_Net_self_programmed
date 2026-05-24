import numpy as np
import torch


def matrix_mult(A,B):
    #print(A.shape,B.shape,"\n")
    #AB = np.einsum('ij,jk->ik', A, B)
    AB = torch.matmul(A,B)
    #import pdb; pdb.set_trace()
    #print(A.shape,B.shape,AB.shape,"\n")
    return AB

"""A=[[1,2,3],[4,5,6]]
B=[[2,3],[4,5],[5,1]]

AB = matrix_mult(A,B)"""


class Model:
    def __init__(self,input_size,output_size):
        hidden_sizes = 256
        self.layers = [#torch.nn.Conv2d(1, 33, 3, stride=1),
                       torch.nn.Conv2d(1,16,3,padding=1),
                       torch.nn.Conv2d(16,32,3,padding=1),
                       torch.randn(512,hidden_sizes, requires_grad=True),
                       torch.randn(hidden_sizes,hidden_sizes, requires_grad=True),
                       torch.randn(hidden_sizes,output_size, requires_grad=True)]
        self.intercepts = [
                           torch.randn(hidden_sizes, requires_grad=True),
                           torch.randn(hidden_sizes, requires_grad=True),
                           torch.randn(output_size, requires_grad=True)]
        for layer in self.layers[2:]:
            #torch.nn.init.xavier_uniform_(layer)
            torch.nn.init.kaiming_uniform_(layer)

        self.params = []

        for layer in self.layers:

            if isinstance(layer, torch.nn.Conv2d):

                self.params.append(layer.weight)
                self.params.append(layer.bias)

            else:
                self.params.append(layer)

        for intercept in self.intercepts:
            self.params.append(intercept)

        self.optimizer = torch.optim.Adam(
            self.params,
            lr=0.001
        )
    

    def activation(self,x):
        #x = torch.relu(x)
        #x = torch.sigmoid(x)
        x = torch.nn.functional.gelu(x)
        return x

    def forward(self,x):

        x  = torch.reshape(x,(x.shape[0],1,8,8))
        for i,layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Conv2d):
                # cnn layer
                x=layer(x)
                if i ==1:
                    x = torch.nn.functional.max_pool2d(x, 2)
                x = self.activation(x)
            else:
                x = x.reshape(x.shape[0], -1)
                intercept = self.intercepts[i-2]
                x = matrix_mult(x,layer) + intercept

                if i < len(self.layers) - 1:

                    #alive = (x > 0).any(dim=0)
                    #dead = (~alive).sum()
                    #print("dead",dead,layer.shape)

                    x = self.activation(x)
        return x

    def train(self,train_data,label_data,batch_size):
        for epoch in range(15):
            print("Epoch:",epoch)
            for i in range(train_data.shape[0]//batch_size):
                preds_x = []
                #for x in train_data[i*batch_size:(i+1)*batch_size]:
                #    pred = self.forward(x)
                #    preds_x.append(pred)
                x = train_data[i*batch_size:(i+1)*batch_size]
                preds_x = self.forward(x)
                #preds_x = torch.stack(preds_x)
                preds_x = preds_x.squeeze(1)

                labels_x = label_data[i*batch_size:(i+1)*batch_size]
                loss = torch.nn.functional.cross_entropy(
                    preds_x,
                    labels_x
                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                

    def test_model(self,x_test,y_test):
        correct = 0
        for i,t in enumerate(x_test):
            pred = model.forward(t)
            if torch.argmax(pred) == y_test[i]:
                correct += 1
        print("test accuracy:",correct/len(x_test))


from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

X = digits.data
y = digits.target
y_new = torch.tensor(y, dtype=torch.long)

def data_preprocess(data):
    data = data.reshape((len(data),1,64))
    data = data / 16.0
    data = torch.tensor(data, dtype=torch.float32)
    return data


new_data = data_preprocess(X)

# train-test-split
x_train = new_data[:-200]
x_test = new_data[-200:]
y_train = y_new[:-200]
y_test = y_new[-200:]
#import pdb; pdb.set_trace()

model = Model(x_train.shape[2],10)

#test
print("test1")
model.test_model(x_test,y_test)

model.train(x_train,y_train,32)
print("model trained")

#test
print("test2")
model.test_model(x_test,y_test)