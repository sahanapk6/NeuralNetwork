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
        print("numlayers", self.num_layers)
        print("layerdims", self.layers_dims)
        print("---------------")

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        A = X
        # print(A)
        A = A.reshape(-1, 1)
        print("input",A)
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
            # print("cache", cache)
            caches.append(cache)

        # print("output",A)
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
            print('params',self.params)
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

                if Y == [[0]]:
                    Y = np.array([[1, 0]])
                elif Y == [[1]]:
                    Y = np.array([[0, 1]])
                print('Y ',Y)
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
                    # print(f'delta_{k}:{dAL[k]}')

                for k in reversed(range(0, self.num_layers)):
                    dWL[k] += np.dot(dAL[k + 1], caches[k][1].T)

            # print('dwL',dWL)
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
            print('params',self.params)

            cost = self.compute_cost(X_test, Y_test)
            cost_list.append(cost)
        return cost_list

    def compute_cost(self, X, Y):
        X = X.to_numpy()
        m = X.shape[0]

        J = 0
        for i in range(m):
            A, c = self.forward_propagation(X[i])
            y = np.array(Y.iloc[i]).reshape((-1, 1))
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
        # print(len(y_true))
        # print(len(y_pred))
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                true_positives += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                # print(y_true[i], i)
                # print(y_pred[i], i)
                false_positives += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                false_negatives += 1
            else:
                true_negatives += 1

        # print("tp",true_positives)
        # print("tn", true_negatives)
        # print("total", len(y_true))
        accuracy = (true_positives + true_negatives) / len(y_true)
        if true_positives + false_positives==0:
            precision=1
            recall=1
        else:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        return accuracy, f1_score
        # return accuracy, precision

    def predict(self, network, X):
        predictions = []
        for i in range(len(X)):
            #print("X.iloc", np.array(X.iloc[i]))
            y_pred, c = network.forward_propagation(np.array(X.iloc[i]))
            #print("predict y_pred", y_pred)
            pred = np.argmax(y_pred)
            #print("predict np.argmax", pred)
            predictions.append(pred)
        return predictions

def one_hot_encode(df):
    mapping = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    for col in df.columns:
        one_hot_col = pd.DataFrame(df[col].apply(lambda x: mapping[x]).tolist(),
                                   columns=[f"{col}_0", f"{col}_1", f"{col}_2"])

        df = pd.concat([df, one_hot_col], axis=1)
        df = df.drop(columns=[col])
    return df

def train_multiple_model(X, Y):
    k = 10
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // k
    accuracy_trainlist = []
    f1_score_trainlist = []
    accuracy_scores = []
    f1_score_scores = []

    hidden_layers_list = [[8], [16], [32], [2,4],[4, 8], [16, 32], [2, 4,8],[4,8,16], [8,16,32],[2,4,8,16],[4,8,8,16],[4,8,16,16]]

    for hidden_layers in hidden_layers_list:
        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.concatenate(
                [indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]

            network = NeuralNetwork(hidden_layers, 48, 2, 1, 1, 400)
            params = network.backward_propagation_withoutcost(X_train, Y_train)

            # print("cost ", params)
            predictions_train = network.predict(network, X_train)
            Y_train = np.array(Y_train).reshape(-1, 1)

            accuracy_train, f1_score_train = network.check_performance(Y_train, predictions_train)
            accuracy_trainlist.append(accuracy_train)
            f1_score_trainlist.append(f1_score_train)

            predictions_test = network.predict(network, X_test)
            # print("pred", len(predictions_test))
            Y_test = np.array(Y_test).reshape(-1, 1)
            # print("Y_test",Y_test.shape)
            accuracy, f1_score = network.check_performance(Y_test, predictions_test)
            accuracy_scores.append(accuracy)
            f1_score_scores.append(f1_score)

        print(hidden_layers)
        print(np.mean(accuracy_trainlist))
        print(np.mean(f1_score_trainlist))
        print(np.mean(accuracy_scores))
        print(np.mean(f1_score_scores))

        print()


def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    network = NeuralNetwork([16, 32], 48, 2, 5, 1, 400)
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
    plt.title('House Votes Dataset :Cost vs Number of iterations')
    plt.show()

def main():
    df = pd.read_csv("house_votes_84.csv")
    X = df.iloc[:, :16]
    Y = df.iloc[:, -1]
    print(Y.iloc[0])
    print(np.array(Y.iloc[0]).reshape((-1, 1)))
    one_hot_df = one_hot_encode(X)
    train_multiple_model(one_hot_df, Y)
    train_model(one_hot_df, Y)

if __name__ == "__main__":
    main()