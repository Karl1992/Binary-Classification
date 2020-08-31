import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2
    y[y == 0] = -1

    w = np.zeros(D)
    # w = np.random.randn(D)
    if w0 is not None:
        w = w0

    b = 0
    # b = np.random.rand(1)
    if b0 is not None:
        b = b0

    w_star = np.append(w, b)
    b_array = np.array([[1] for _ in range(N)])
    X = np.append(X, b_array, axis=1)
    Z = y * np.dot(w_star, X.T)

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        # def core_label(core1):
        #     for i in range(len(core1)):
        #         if core1[i] < 0:
        #             core1[i] = 0
        #     # perceptron_loss1 = sum(core1)
        #     return core1

        def Z_label(Z):
            Z = np.array([1 if Zi <= 0 else 0 for Zi in Z])
            # for i in range(len(Z)):
            #     if Z[i] <= 0:
            #         Z[i] = 1
            #     else:
            #         Z[i] = 0
            return Z

        # core, perceptron_loss = core_label(-Z)

        this_Z = Z
        this_w_star = w_star
        # best_w_star = this_w_star
        iter_times = 0
        while iter_times < max_iterations:
            iter_times += 1
            this_w_star = this_w_star + np.dot(step_size * Z_label(this_Z) * y, X) / N
            this_Z = y * np.dot(this_w_star, X.T)
            # this_core, this_perceptron_loss = core_label(-this_Z)
            # if this_perceptron_loss <= perceptron_loss:
            #     perceptron_loss = this_perceptron_loss
            #     best_w_star = this_w_star

        w = this_w_star[:-1]
        b = this_w_star[-1]
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        # core = -Z
        # logistic_loss = sum(np.log(1 + np.exp(core)))
        this_w_star = w_star
        iter_times = 0
        # best_w_star = this_w_star

        while iter_times < 1000:
            iter_times += 1
            P = sigmoid(-Z)
            this_w_star = this_w_star + np.dot(step_size / N * P * y, X)
            Z = y * np.dot(this_w_star, X.T)
            # this_logistic_loss = sum(np.log(1 + np.exp(core)))

            # if this_logistic_loss <= logistic_loss:
            #     logistic_loss = this_logistic_loss
            #     best_w_star = this_w_star

        w = this_w_star[:-1]
        b = this_w_star[-1]
        ############################################


    else:
        raise "Loss Function is undefined."

    y[y == -1] = 0
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.dot(w, X.T) + b
        preds[preds > 0] = 1
        preds[preds <= 0] = 0
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = sigmoid(np.dot(w, X.T) + b)
        preds = np.array([1 if y_hat > 0.5 else 0 for y_hat in preds])
        ############################################


    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    def core_calculator(m):
        m = np.exp(m)
        denominator = np.sum(m, axis=0)
        return np.divide(m, denominator)

    # def core_calculator_gd(W, x, c_matrix):
    #     return np.exp(np.dot(W, x.T) - sum((np.dot(c_matrix.T, W) * x).T))

    def W_matrix_renew(this_w_star, this_n, x, y, step_size, C, D, p_matrix):
        p_matrix[y[this_n]] = p_matrix[y[this_n]] - 1
        this_w_star = this_w_star - np.dot(np.multiply(step_size, p_matrix.reshape((C, 1))), x[this_n].reshape((1, D + 1)))
        return this_w_star

    def W_matrix_renew_gd(this_w_star, x, step_size, p_matrix, N):
        this_w_star = this_w_star - np.divide(np.dot(step_size * p_matrix, x), N)
        return this_w_star

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    w_star = np.vstack((w.T, b)).T
    b_array = np.array([[1] for _ in range(len(X))])
    X = np.append(X, b_array, axis=1)

    # max_X = np.max(X)
    # new_X = X - max_X

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        this_w_star = w_star
        iter_times = 0
        while iter_times < max_iterations:
            iter_times += 1
            this_n = np.random.choice(N)
            this_X = X[this_n]
            this_core = core_calculator(np.dot(this_w_star, this_X.T))
            this_w_star = W_matrix_renew(this_w_star, this_n, X, y, step_size, C, D, this_core)

        w = this_w_star[:, :-1]
        b = this_w_star[:, -1]
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        c_matrix = np.eye(C)[y]
        # for i in range(N):
        #     c_matrix[y[i], i] = 1
        this_w_star = w_star
        # best_w_star = w_star
        # this_core = core_calculator_gd(np.dot(this_w_star, X.T)) - c_matrix.T
        # log_loss = sum(np.log(sum(this_core)))
        iter_times = 0
        while iter_times < max_iterations:
            iter_times += 1
            this_core = core_calculator(np.dot(this_w_star, X.T)) - c_matrix.T
            this_w_star = W_matrix_renew_gd(this_w_star, X, step_size, this_core, N)
            # this_log_loss = sum(np.log(sum(this_core)))
            # if this_log_loss <= log_loss:
            #     log_loss = this_log_loss
            #     best_w_star = this_w_star

        w = this_w_star[:, :-1]
        b = this_w_star[:, -1]
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    # C = b.shape
    y = np.dot(w, X.T).T + b  # * np.ones(C)#
    preds = np.argmax(y, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds