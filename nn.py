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

class Loss():
    def __innit__(name):
        loss_function = name
        return loss_function
    def loss(preds,targets):
        loss = sum((preds-targets)**2)
        return loss


class Model:
    def __init__(self,input_size,output_size):
        hidden_sizes = 256
        self.layers = [torch.randn(input_size,hidden_sizes, requires_grad=True),
                       torch.randn(hidden_sizes,hidden_sizes, requires_grad=True),
                       torch.randn(hidden_sizes,output_size, requires_grad=True)]
        self.intercepts = [torch.randn(hidden_sizes, requires_grad=True),
                           torch.randn(hidden_sizes, requires_grad=True),
                           torch.randn(output_size, requires_grad=True)]

    def Relu(self,x):
        x = torch.relu(x)
        return x

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            intercept = self.intercepts[i]
            x = matrix_mult(x,layer) + intercept
            if i < len(self.layers) - 1:
                x = model.Relu(x)
        return x

    def train(self,train_data,label_data,batch_size):
        for epoch in range(15):
            print("Epoch:",epoch)
            for i in range(train_data.shape[0]//batch_size):
                preds_x = []
                #for x in train_data[i*batch_size:(i+1)*batch_size]:
                #    pred = self.forward(x)
                #    preds_x.append(pred)
                preds_x = self.forward(train_data[i*batch_size:(i+1)*batch_size])
                #preds_x = torch.stack(preds_x)
                preds_x = preds_x.squeeze(1)

                labels_x = label_data[i*batch_size:(i+1)*batch_size]
                loss = torch.nn.functional.cross_entropy(
                    preds_x,
                    labels_x
                )
                loss.backward()

                learning_rate = 0.01
                with torch.no_grad():
                    for i,layer in enumerate(self.layers):
                        intercept = self.intercepts[i]
                        layer -= learning_rate * layer.grad
                        layer.grad.zero_()
                        intercept -= learning_rate * intercept.grad
                        intercept.grad.zero_()

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