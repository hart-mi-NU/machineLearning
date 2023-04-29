
from sklearn.model_selection import train_test_split
import numpy as np

# Do not change the code in this cell
true_slope = 15
true_intercept = 2.4
input_var = np.arange(0.0, 100.0)
output_var = true_slope * input_var + true_intercept + \
    300.0 * np.random.rand(len(input_var))


def compute_cost(ip, op, params):
    """
    Cost function in linear regression where the cost is calculated
    ip: input variables
    op: output variables
    params: corresponding parameters
    Returns cost
    """
    num_samples = len(ip)
    cost_sum = 0.0
    for x, y in zip(ip, op):
        y_hat = np.dot(params, np.array([1.0, x]))
        cost_sum += (y_hat - y) ** 2

    cost = cost_sum / (num_samples)

    return cost


def updateParams(params, alpha, ip, op):
    '''
    Updates the params vector
    '''
    num_samples = len(ip)

    sumErrorsW0 = 0.0
    for x, y in zip(ip, op):
        y_hat = np.dot(params, np.array([1.0, x]))
        sumErrorsW0 += (y_hat - y)
    print("sumError = " + str(sumErrorsW0))

    sumErrorsW1 = 0.0
    for x, y in zip(ip, op):
        y_hat = np.dot(params, np.array([1.0, x]))
        sumErrorsW1 += ((y_hat - y))*x
    # print("sumErrorX = " + str(sumErrorsW1))

    params[0] = params[0] - alpha*(sumErrorsW0)/num_samples
    params[1] = params[1] - alpha*(sumErrorsW1)/num_samples
    print("params = " + str(params))
    print("-----")


def linear_regression_using_batch_gradient_descent(ip, op, params, alpha, max_iter):
    """
    Compute the params for linear regression using batch gradient descent
    ip: input variables
    op: output variables
    params: corresponding parameters
    alpha: learning rate
    max_iter: maximum number of iterations
    Returns parameters, cost, params_store
    """
    # initialize iteration, number of samples, cost and parameter array
    iteration = 0
    num_samples = len(ip)
    cost = np.zeros(max_iter)
    params_store = np.zeros([2, max_iter])  # two arrays, each with 100 zeros

    # Compute the cost and store the params for the corresponding cost
    while iteration < max_iter:
        # get the SSE for the current set of params
        cost[iteration] = compute_cost(ip, op, params)
        # save the current params in the param store
        params_store[:, iteration] = params

        print('--------------------------')
        print(f'iteration: {iteration}')
        print(f'cost: {cost[iteration]}')

        # Apply batch gradient descent

        # Update params
        updateParams(params, alpha, ip, op)

        iteration += 1

    return params, cost, params_store


def updateParamsStochastic(params, alpha, x, y, i):
    '''
    Updates the params vector via stochastic method
    '''
    m = len(x)
    y_hat = np.dot(params, np.array([1.0, x[i]]))
    error = y_hat - y[i]
    params[0] = params[0] - (alpha * error / m)
    params[1] = params[1] - (alpha * error / m * x[i])


def lin_reg_stoch_gradient_descent(ip, op, params, alpha):
    """
    Compute the params for linear regression using stochastic gradient descent
    ip: input variables
    op: output variables
    params: corresponding parameters
    alpha: learning rate
    Returns parameters, cost, params_store
    """

    # initialize iteration, number of samples, cost and parameter array
    num_samples = len(input_var)
    cost = np.zeros(num_samples)
    params_store = np.zeros([2, num_samples])

    # run SGD for a set number of epochs
    max_epochs = 100
    epoch = 0
    while epoch < max_epochs:
        i = 0
        # Compute the cost and store the params for the corresponding cost
        for x, y in zip(input_var, output_var):
            cost[i] = compute_cost(input_var, output_var, params)
            params_store[:, i] = params

            print('--------------------------')
            print(f'iteration: {i}')
            print(f'cost: {cost[i]}')

            # Apply stochastic gradient descent
            numIters = 50
            updateParamsStochastic(params, alpha, input_var, output_var, i)
            i += 1  # i will increment until i == len(op)

        epoch += 1

    return params, cost, params_store


x_train, x_test, y_train, y_test = train_test_split(
    input_var, output_var, test_size=0.20)

params_0 = np.array([20.0, 80.0])

alpha_batch = 1e-4
max_iter = 200
params_hat_batch, cost_batch, params_store_batch =\
    lin_reg_stoch_gradient_descent(
        x_train, y_train, params_0, alpha_batch)

print(params_hat_batch)
