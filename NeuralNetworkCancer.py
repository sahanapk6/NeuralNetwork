import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, hidden_dims, input_dim, num_classes, regularization, learning_rate, iterations):
        self.regularization = float(regularization)
        self.learning_rate = learning_rate
        self.num_layers = 1 + len(hidden_dims)
        self.params = []
        self.params = [np.array(param) for param in self.params]
        layers_dims = [input_dim] + hidden_dims + [num_classes]
        self.layers_dims = layers_dims
        self.iterations=iterations

        for i in range(self.num_layers):
            #W = np.random.randn(layers_dims[i + 1], layers_dims[i] + 1)
            W = np.random.uniform(low=-1.0, high=1.0, size=(layers_dims[i + 1], layers_dims[i] + 1))
            self.params.append(W)
        #     print("param", i, self.params[i])
        #
        # print("numlayers", self.num_layers)
        # print("layerdims", self.layers_dims)
        # self.params.append(np.array([[0.13530023, 0.77811384, 0.61898355 , 0.18781898, -0.45935115, -0.44273603,-0.56549639, -0.61883665, 0.55439558,-0.7496471,  0.61557101,0.73233756,0.79088185, -0.31093712],
        #  [0.3961599, -0.9780375, -0.37936045,  0.34951994, -0.02074869, -0.88681414, 0.96393461, -0.97825591, -0.14483057, -0.66764416, -0.68941265,  0.14080814,0.55759838, -0.01002002]]))
        #
        # self.params.append(np.array([[-0.38519238,  0.27229991,  0.50382974],
        #  [0.43913652, -0.9782524, -0.9526196],
        #  [0.66171218,  0.71211977,  0.00410946],
        #  [-0.40428296, -0.63002383, -0.8197302]]))
        #
        #
        # self.params.append(np.array([[0.18194689,  0.82036619,  0.60663884,  0.36326435,  0.84065459],
        #  [0.958001,    0.03190983, -0.20221481, -0.89815541, -0.26532255],
        # [0.73904321, -0.32939187, 0.68285224, -0.85426886, -0.39008302]]))
        # print("---------------")

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        A = X
        # print(A)
        A = A.reshape(-1, 1)
        # print("input",A)
        caches = []
        print("-----------------forward_propagation--------------")
        print("A1=", A)
        for i in range(self.num_layers):
            # compute activation of current layer
            W = self.params[i]
            print("W", i + 1, "=", W)
            A = np.vstack(([1], A))
            Z = np.dot(W, A)
            print("Z", i + 2, "=", Z)
            A_prev = A
            A = self.sigmoid(Z)
            print("A", i + 2, "=", A)
            cache = (W, A_prev, A)
            print("cache", cache)
            caches.append(cache)
        #
        return A, caches

    def backward_propagation_withoutcost(self, X_train, Y_train):
        m = len(X_train)
        dWL = []
        cost_list = []
        for _ in range(self.iterations):
            for i in range(len(self.layers_dims) - 1):
                dWL.append(np.zeros((self.layers_dims[i + 1], self.layers_dims[i] + 1)))
            for i in range(len(X_train)):
                dAL = [0 for _ in range(self.num_layers + 1)]
                X = np.array(X_train.iloc[i])
                Y = np.array(Y_train.iloc[i])
                # print("seeeeeeee ", X, Y)
                if Y == [[0]]:
                    Y = np.array([[1, 0]])
                elif Y == [[1]]:
                    Y = np.array([[0, 1]])

                # print('Y ',Y)
                print("-----------------backward_propagation--------------")
                print(f"-----------------instance = {i}--------------")
                A, caches = self.forward_propagation(X)
                Y = Y.reshape(-1, 1)
                delta = A - Y
                print(f'delta_{self.num_layers}:{delta}')
                dAL[-1] = delta
                delta_prev = delta
                for k in reversed(range(0, self.num_layers)):
                    current_cache = caches[k]
                    W = current_cache[0]
                    A_prev = current_cache[1]
                    A = current_cache[2]
                    W = np.array(W, ndmin=2)

                    dAL[k] = np.dot(W.T, dAL[k + 1]) * A_prev * (1 - A_prev)
                    dAL[k] = dAL[k][1:]
                    dAL[k] = dAL[k].reshape(-1, 1)
                    print(f'delta_{k}:{dAL[k]}')

                for k in reversed(range(0, self.num_layers)):
                    dWL[k] += np.dot(dAL[k + 1], caches[k][1].T)

            print('dwL',dWL)
            print()

            P = []
            D = []
            for k in range(self.num_layers):
                W = self.params[k]
                reg = self.regularization * W
                reg[:, 0] = 0
                P.append(reg)
                dWL[k] = (1 / m) * (dWL[k] + P[k])

            for k in range(self.num_layers):
                self.params[k] -= self.learning_rate * dWL[k]
            # print('params',self.params)

        return 1

    def backward_propagation(self, X_train, Y_train, X_test, Y_test):
        m = len(X_train)
        dWL = []
        cost_list = []
        for _ in range(self.iterations):
            for i in range(len(self.layers_dims) - 1):
                dWL.append(np.zeros((self.layers_dims[i + 1], self.layers_dims[i] + 1)))
            for i in range(len(X_train)):
                dAL = [0 for _ in range(self.num_layers + 1)]
                X = np.array(X_train.iloc[i])
                Y = np.array(Y_train.iloc[i])
                # print("seeeeeeee ", X, Y)
                if Y == [[0]]:
                    Y = np.array([[1, 0]])
                elif Y == [[1]]:
                    Y = np.array([[0, 1]])

                # print('Y ',Y)
                print("-----------------backward_propagation--------------")
                print(f"-----------------instance = {i}--------------")
                A, caches = self.forward_propagation(X)
                Y = Y.reshape(-1, 1)
                delta = A - Y
                print(f'delta_{self.num_layers}:{delta}')
                dAL[-1] = delta
                delta_prev = delta
                for k in reversed(range(0, self.num_layers)):
                    current_cache = caches[k]
                    W = current_cache[0]
                    A_prev = current_cache[1]
                    A = current_cache[2]
                    W = np.array(W, ndmin=2)

                    dAL[k] = np.dot(W.T, dAL[k + 1]) * A_prev * (1 - A_prev)
                    dAL[k] = dAL[k][1:]
                    dAL[k] = dAL[k].reshape(-1, 1)
                    print(f'delta_{k}:{dAL[k]}')

                for k in reversed(range(0, self.num_layers)):
                    dWL[k] += np.dot(dAL[k + 1], caches[k][1].T)

            print('dwL',dWL)
            print()

            P = []
            D = []
            for k in range(self.num_layers):
                W = self.params[k]
                reg = self.regularization * W
                reg[:, 0] = 0
                P.append(reg)
                dWL[k] = (1 / m) * (dWL[k] + P[k])

            for k in range(self.num_layers):
                self.params[k] -= self.learning_rate * dWL[k]
            # print('params',self.params)

            cost = self.compute_cost(X_test, Y_test)
            cost_list.append(cost)
        return cost_list

    def compute_cost(self, X, Y):
        X = np.array(X)
        m = X.shape[0]

        J = 0
        for i in range(m):
            A, c = self.forward_propagation(X[i])
            y = np.array(Y.iloc[i]).reshape((-1, 1))
            if y == [[0]]:
                y = np.array([[1, 0]])
            elif y == [[1]]:
                y = np.array([[0, 1]])
            j = - y * np.log(A) - (1 - y) * np.log(1 - A)
            J += np.sum(j)
            # print("cost of instance ", i, "=", np.sum(j))
        J = J / m

        S = 0
        for param in self.params:
            S += np.sum(np.square(param[:, 1:]))
        S = (self.regularization / (2 * m)) * S
        # print(S)
        J += S
        return J

    def check_performance(self, y_true, y_pred):
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(y_true)):
            if y_true[i] == 0 and y_pred[i] == 0:
                true_positives += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                false_positives += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                false_negatives += 1
            else:
                true_negatives += 1

        # print("tp",true_positives)
        # print("tn", true_negatives)
        # print("total ", len(y_true))
        accuracy = (true_positives + true_negatives) / len(y_true)
        if true_positives + false_positives==0:
            precision=1
            recall=1
        else:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        return accuracy,f1_score

    def predict(self, network, X):
        predictions = []
        for i in range(len(X)):
            y_pred, c = network.forward_propagation(np.array(X.iloc[i]))
            pred = np.argmax(y_pred)
            # print("predict np.argmax", pred+1)
            predictions.append(pred)
        return predictions

def train_multiple_model(X, Y):

    k = 10
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // k
    accuracy_trainlist = []
    f1_score_trainlist = []
    accuracy_scores = []
    f1_score_scores = []

    hidden_layers_list = [[8], [16], [32], [2, 4], [4, 8], [16, 32], [2, 4, 8], [4, 8, 16], [8, 16, 32], [2, 4, 8, 16],
                          [4, 8, 8, 16], [4, 8, 16, 16],[8, 8, 16, 16],[2,4,8, 16, 32]]


    for hidden_layers in hidden_layers_list:
        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.concatenate(
                [indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]

            network = NeuralNetwork(hidden_layers, 9, 2, 1, 1, 500)
            params = network.backward_propagation_withoutcost(X_train, Y_train)

            predictions_train = network.predict(network, X_train)
            Y_train = np.array(Y_train).reshape(-1, 1)
            accuracy_train, f1_score_train = network.check_performance(Y_train, predictions_train)
            accuracy_trainlist.append(accuracy_train)
            f1_score_trainlist.append(f1_score_train)

            predictions = network.predict(network, X_test)
            Y_test = np.array(Y_test).reshape(-1, 1)
            accuracy, f1_score = network.check_performance(Y_test, predictions)
            accuracy_scores.append(accuracy)
            f1_score_scores.append(f1_score)

        print(hidden_layers)
        print(np.mean(accuracy_trainlist))
        print(np.mean(f1_score_trainlist))
        print(np.mean(accuracy_scores))
        print(np.mean(f1_score_scores))

        print()

def train_model(X, Y):
    shuffled_indices = np.random.permutation(len(X))
    X = X.iloc[shuffled_indices]
    Y = Y.iloc[shuffled_indices]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    network = NeuralNetwork([4, 8], 9, 2, 5, 1, 500)
    cost = network.backward_propagation(X_train, Y_train, X_test, Y_test)
    print("cost ", cost)
    predictions_train = network.predict(network, X_train)
    Y_train = np.array(Y_train).reshape(-1, 1)
    accuracy_train, f1_score_train = network.check_performance(Y_train, predictions_train)
    print("Train Accuracy:", accuracy_train)
    print("Train F1 Score:", f1_score_train)
    predictions_test = network.predict(network, X_test)
    Y_test = np.array(Y_test).reshape(-1, 1)
    accuracy_test, f1_score_test = network.check_performance(Y_test, predictions_test)
    print("Test Accuracy:", accuracy_test)
    print("Test F1 Score:", f1_score_test)

    plt.plot(range(len(cost)), cost)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.title('Cancer Dataset :Cost vs Number of iterations')
    plt.show()

def normalize(dataset):
    for col in dataset.columns:
        dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())
    return dataset

def main():
    df = pd.read_csv("cancer.csv", sep='\t')
    X = df.iloc[:, :9]
    Y = df.iloc[:, -1]
    # print(X.head())
    # print(Y.head())
    X_normalized = normalize(X)
    train_multiple_model(X_normalized, Y)
    train_model(X_normalized, Y)
    # print(acc)

if __name__ == "__main__":
    main()