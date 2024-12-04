import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import locale
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')


data_test = pd.read_csv('test.csv')

X_test = data_test.iloc[:, 1:].values.T / 255.0
Y_test = None


# Attempt to read the CSV file for the table of 5 rows and 785 columns for pixels
data = pd.read_csv('train.csv')

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Split into development and training sets
data_dev = data[0:1000]
data_train = data[1000:]

#.iloc for integer-based indexing
X_dev = data_dev.iloc[:, 1:].values.T / 255.
Y_dev = data_dev.iloc[:, 0].values

X_train = data_train.iloc[:, 1:].values.T / 255.
Y_train = data_train.iloc[:, 0].values
_, m_train = X_train.shape

train_df = pd.DataFrame(X_train.T)
dev_df = pd.DataFrame(X_test.T)

# Check for duplicates
duplicates = train_df.merge(dev_df, how='inner')

if not duplicates.empty:
    print("Data leakage detected: Training and development sets have overlapping samples.")
    print("Overlapping Samples:")
    print(duplicates)
    print(f"Number of Overlapping Samples: {len(duplicates)}")
else:
    print("No overlap detected between training and development sets.")


def init_params():
    W_input_hidden1 = np.random.randn(64, 784) * np.sqrt(2. / 784)
    b_input_hidden1 = np.zeros((64, 1))

    W_hidden1_hidden2 = np.random.randn(32, 64) * np.sqrt(2. / 64)
    b_hidden1_hidden2 = np.zeros((32, 1))

    W_hidden2_output = np.random.randn(10, 32) * np.sqrt(2. / 32)
    b_hidden2_output = np.zeros((10, 1))

    return W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output



def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtract max for numerical stability
    return A / np.sum(A, axis=0, keepdims=True)


def forward_prop(W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output,
                 b_hidden2_output, X):
    # First hidden layer
    Z_input_hidden1 = W_input_hidden1.dot(X) + b_input_hidden1
    A_hidden1 = ReLU(Z_input_hidden1)

    # Second hidden layer
    Z_hidden1_hidden2 = W_hidden1_hidden2.dot(A_hidden1) + b_hidden1_hidden2
    A_hidden2 = ReLU(Z_hidden1_hidden2)

    # Output layer
    Z_hidden2_output = W_hidden2_output.dot(A_hidden2) + b_hidden2_output
    A_output = softmax(Z_hidden2_output)

    return Z_input_hidden1, A_hidden1, Z_hidden1_hidden2, A_hidden2, Z_hidden2_output, A_output


def one_hot(Y, num_classes=10):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y.astype(int), np.arange(Y.size)] = 1
    return one_hot_Y


def deriv_ReLU(Z):
    return Z > 0


def backward_prop(Z_input_hidden1, A_hidden1, Z_hidden1_hidden2, A_hidden2, Z_hidden2_output, A_output,
                  W_input_hidden1, W_hidden1_hidden2, W_hidden2_output, X, Y, lambd):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    # Gradients for output layer
    dZ_hidden2_output = A_output - one_hot_Y
    dW_hidden2_output = (1 / m) * dZ_hidden2_output.dot(A_hidden2.T) + (lambd / m) * W_hidden2_output
    db_hidden2_output = (1 / m) * np.sum(dZ_hidden2_output, axis=1, keepdims=True)

    # Gradients for second hidden layer
    dZ_hidden1_hidden2 = W_hidden2_output.T.dot(dZ_hidden2_output) * deriv_ReLU(Z_hidden1_hidden2)
    dW_hidden1_hidden2 = (1 / m) * dZ_hidden1_hidden2.dot(A_hidden1.T) + (lambd / m) * W_hidden1_hidden2
    db_hidden1_hidden2 = (1 / m) * np.sum(dZ_hidden1_hidden2, axis=1, keepdims=True)

    # Gradients for first hidden layer
    dZ_input_hidden1 = W_hidden1_hidden2.T.dot(dZ_hidden1_hidden2) * deriv_ReLU(Z_input_hidden1)
    dW_input_hidden1 = (1 / m) * dZ_input_hidden1.dot(X.T) + (lambd / m) * W_input_hidden1
    db_input_hidden1 = (1 / m) * np.sum(dZ_input_hidden1, axis=1, keepdims=True)

    return dW_input_hidden1, db_input_hidden1, dW_hidden1_hidden2, db_hidden1_hidden2, dW_hidden2_output, db_hidden2_output


def update_params(W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output,
                  b_hidden2_output,
                  dW_input_hidden1, db_input_hidden1, dW_hidden1_hidden2, db_hidden1_hidden2, dW_hidden2_output,
                  db_hidden2_output, alpha):
    W_input_hidden1 = W_input_hidden1 - alpha * dW_input_hidden1
    b_input_hidden1 = b_input_hidden1 - alpha * db_input_hidden1

    W_hidden1_hidden2 = W_hidden1_hidden2 - alpha * dW_hidden1_hidden2
    b_hidden1_hidden2 = b_hidden1_hidden2 - alpha * db_hidden1_hidden2

    W_hidden2_output = W_hidden2_output - alpha * dW_hidden2_output
    b_hidden2_output = b_hidden2_output - alpha * db_hidden2_output

    return W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output


def get_predictions(A_output):
    return np.argmax(A_output, 0)


def get_accuracy(predictions, Y):
    print("Sample predictions:", predictions[:10])
    print("Sample true labels:", Y[:10])
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha, batch_size, lambd):
    W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output = init_params()
    m = X.shape[1]

    for i in range(iterations):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]

        #stochastic gradient descent (SGD)
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]

            # Forward and backward propagation on the batch
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(
                W_input_hidden1, b_input_hidden1,
                W_hidden1_hidden2, b_hidden1_hidden2,
                W_hidden2_output, b_hidden2_output, X_batch  # Use X_batch here
            )
            # Backward propagation with L2 regularization
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(
                Z1, A1, Z2, A2, Z3, A3,
                W_input_hidden1, W_hidden1_hidden2, W_hidden2_output, X_batch, Y_batch, lambd  # Use X_batch and Y_batch
            )
            # Update parameters
            W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output = update_params(
                W_input_hidden1, b_input_hidden1,
                W_hidden1_hidden2, b_hidden1_hidden2,
                W_hidden2_output, b_hidden2_output,
                dW1, db1, dW2, db2, dW3, db3, alpha
            )

        if (i % 10 == 0):
            # Evaluate on the dev set
            _, _, _, _, _, A_dev = forward_prop(W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2,
                                                W_hidden2_output, b_hidden2_output, X_dev)
            dev_predictions = get_predictions(A_dev)
            dev_accuracy = get_accuracy(dev_predictions, Y_dev)
            print(f"Iteration: {i}, Dev Accuracy: {dev_accuracy}")

    return W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output





def make_predictions(X, W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output):
    _, _, _, _, _, A_output = forward_prop(W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output, X)
    predictions = get_predictions(A_output)
    return predictions


def test_prediction(index, W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2,
                    W_hidden2_output, b_hidden2_output):
    print(f"Testing index: {index}")
    current_image = X_test[:, index]

    prediction = make_predictions(current_image.reshape(-1, 1), W_input_hidden1, b_input_hidden1,
                                  W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output)
    print("Prediction: ", prediction[0])

    label = Y_test[index]
    print("Actual Label: ", label)

    image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
 W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2, W_hidden2_output, b_hidden2_output = gradient_descent(
    X_train, Y_train, iterations=10, alpha=0.1, batch_size=64,  lambd=0.01)
 while True:
    user_input = input(f"Enter an image index to test (between 0 and {X_test.shape[1] - 1}): ")
    print(f"Raw input received: {user_input!r}")

    test = int(user_input.strip())

    if test == -1:
        print("Exiting program")
        break
    elif 0 <= test < X_test.shape[1]:
        try:
            test_prediction(test, W_input_hidden1, b_input_hidden1, W_hidden1_hidden2, b_hidden1_hidden2,
                            W_hidden2_output, b_hidden2_output)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
