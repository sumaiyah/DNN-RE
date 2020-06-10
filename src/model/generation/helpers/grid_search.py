from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from src import NN_INIT_GRID_RESULTS_FP
from model.generation.helpers.build_and_train_model import create_model


def grid_search(X, y):
    """

    Args:
        X: input features
        y: target

    Returns:
        batch_size: best batch size
        epochs: best number of epochs
        layer_1: best number of neurons for layer 1 (first hidden layer)
        layer_2: best number of neurons for layer 2

    Perform a 5-folded grid search over the neural network hyper-parameters
    """
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 400]
    layer_1 = [10, 15, 50, 100]
    layer_2 = [2, 5, 10, 50]

    param_grid = dict(batch_size=batch_size,
                      epochs=epochs,
                      layer_1=layer_1,
                      layer_2=layer_2)

    model = KerasClassifier(build_fn=create_model, verbose=0)

    print('hi1')
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, verbose=10)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=10)
    print('hi2')
    grid_result = grid.fit(X, y)
    print('hi3')

    # Write best results to file
    with open(NN_INIT_GRID_RESULTS_FP, 'w') as file:
        file.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            file.write("%f (%f) with: %r\n" % (mean, stdev, param))

    print('Grid Search for hyper parameters complete.')
    print("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))

    return grid_result.best_params_['batch_size'], grid_result.best_params_['epochs'], \
           grid_result.best_params_['layer_1'], grid_result.best_params_['layer_2']