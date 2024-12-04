import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print("a")

# Attempt to read the CSV file for the table of 5 rows and 785 columns for pixels
data = pd.read_csv('train.csv')

data = np.array(data)  # Convert to numpy array
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params():
    W_input_hidden = np.random.randn(64, 784)
    b_input_hidden = np.random.randn(64, 1)
    W_hidden_output = np.random.randn(10, 64)
    b_hidden_output = np.random.randn(10, 1)
    return W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtract max for numerical stability
    return A / np.sum(A, axis=0, keepdims=True)


def forward_prop(W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output, X):
    Z_input_hidden = W_input_hidden.dot(X) + b_input_hidden  # Add bias
    A_hidden = ReLU(Z_input_hidden)
    Z_hidden_output = W_hidden_output.dot(A_hidden) + b_hidden_output  # Add bias
    A_output = softmax(Z_hidden_output)  # Use softmax for output layer
    return Z_input_hidden, A_hidden, Z_hidden_output, A_output


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def deriv_ReLU(Z):
    return Z > 0


def backward_prop(Z_input_hidden, A_hidden, Z_hidden_output, A_output, W_input_hidden, W_hidden_output, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ_hidden_output = A_output - one_hot_Y
    dW_hidden_output = 1e-04 * dZ_hidden_output.dot(A_hidden.T)
    db_hidden_output = 1e-04 * np.sum(dZ_hidden_output)

    dZ_input_hidden = W_hidden_output.T.dot(dZ_hidden_output) * deriv_ReLU(Z_input_hidden)
    dW_input_hidden = 1e-04 * dZ_input_hidden.dot(X.T)
    db_input_hidden = 1e-04 * np.sum(dZ_input_hidden)
    return dW_input_hidden, db_input_hidden, dW_hidden_output, db_hidden_output


def update_params(W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output, dW_input_hidden, db_input_hidden,
                  dW_hidden_output, db_hidden_output, alpha):
    W_input_hidden = W_input_hidden - alpha * dW_input_hidden
    b_input_hidden = b_input_hidden - alpha * db_input_hidden
    W_hidden_output = W_hidden_output - alpha * dW_hidden_output
    b_hidden_output =  b_hidden_output - alpha * db_hidden_output
    return W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output


def get_predictions(A_output):
    return np.argmax(A_output, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W_input_hidden, W_hidden_output, X, Y)
        W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output = update_params(
            W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output, dW1, db1, dW2, db2, alpha
        )
        if (i % 100 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output




def make_predictions(X, W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output):
    _, _, _, A_output = forward_prop(W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output, X)
    predictions = get_predictions(A_output)
    return predictions


def test_prediction(index, W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output):
    current_image = X_train[:, index]
    prediction = make_predictions(X_train[:, index, None], W_input_hidden, b_input_hidden, W_hidden_output,
                                  b_hidden_output)


    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



if __name__ == "__main__":
    W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output = gradient_descent(X_train, Y_train, 1000, 0.1)
    while True:
        test = int(input("Number of test images: "))
        test_prediction(test, W_input_hidden, b_input_hidden, W_hidden_output, b_hidden_output)
