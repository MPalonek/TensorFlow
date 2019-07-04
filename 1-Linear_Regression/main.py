# https://drive.google.com/drive/folders/1_ZMBQAXYjg1KNKErdWv9Ne2XOThkZnR0?usp=sharing

from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# initial setup
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def my_input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # convert pandas data into a dict of np arrays
    features = {key:np.array(value) for key,value in dict(features).items()}

    # construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets)) # warning 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, my_dataframe, input_feature="total_rooms"):
    """Trains a linear regression model of one feature.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = my_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = my_dataframe[my_label]

    # create feature columns
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # create input functions
    training_input_fn = lambda: my_input_function(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_function(my_feature_data, targets, shuffle=False, num_epochs=1)

    # create a linear regressor object
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

    # set up to plot the state of our models line each period
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line By Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = my_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # train the model, but do so inside a loop so that we can periodically assess loss metrics
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range (0, periods):
        # train the model, starting from the prior state
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        # take a break and compute predictions
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # compute loss
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
        # occasionally print the current loss
        print(" period %02d : %0.2f" % (period, root_mean_squared_error))
        # add the loss metrics from this period to our list
        root_mean_squared_errors.append(root_mean_squared_error)
        # finally, track the weights and biases over time
        # apply some math to ensure that the data and line are plotted neatly
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[my_feature].max()), sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print("Model training finished")

    # output a graph of loss metrics over periods
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # output a table with calibration data
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
    plt.show()


def main():
    # load data set
    california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

    # randomize the data and scale median_house_value to be in thousands
    california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
    california_housing_dataframe["median_house_value"] /= 1000.0

    # print out few useful statistics on each column
    print(california_housing_dataframe.describe())

    # define the input feature: total_rooms
    my_feature = california_housing_dataframe[["total_rooms"]]

    # configure a numeric feature column for total_rooms (in tf data can be either categorical or numeric)
    feature_column = [tf.feature_column.numeric_column("total_rooms")]

    # define the label
    targets = california_housing_dataframe["median_house_value"]

    # use gradient descent as the optimizer for training the model
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    # configure the linear regression model with our feature column and optimizer.
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_column, optimizer=my_optimizer)

    # train the model
    linear_regressor.train(input_fn=lambda:my_input_function(my_feature, targets), steps=100)

    # create an input function for predictions
    # Note: since we are making just one prediction for each example, we dont need to repeat or shuffle data here
    prediction_input_fn = lambda: my_input_function(my_feature, targets, num_epochs=1, shuffle=False)

    # call predict() on the linear_regressor to make predictions
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)

    # format predictions as a numpy array, so we can calculate error metrics
    predictions = np.array([item['predictions'][0] for item in predictions])

    # print mean squared error and root mean squared error
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
    print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

    # compare RMSE to difference of min and max of our targets to judge model error
    min_house_value = california_housing_dataframe["median_house_value"].min()
    max_house_value = california_housing_dataframe["median_house_value"].max()
    min_max_difference = max_house_value - min_house_value

    print("Min. Median House Value: %0.3f" % min_house_value)
    print("Max. Median House Value: %0.3f" % max_house_value)
    print("Difference between Min. and Max.: %0.3f" % min_max_difference)
    print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

    # take a look at how well our predictions matches our targets
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    print(calibration_data.describe())

    # in next steps we will visualize the data and our model prediction
    # get uniform random sample of the data
    sample = california_housing_dataframe.sample(n=300)
    # get the min and max total_rooms values
    x_0 = sample["total_rooms"].min()
    x_1 = sample["total_rooms"].max()
    # retrieve the final weight and bias generated during training
    weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    # get the predicted median_house_values for the min and max total_rooms values
    y_0 = weight * x_0 + bias
    y_1 = weight * x_1 + bias
    # plot our regression line from (x_0, y_0) to (x_1, y_1)
    plt.plot([x_0, x_1], [y_0, y_1], c='r')
    # label the graph axes
    plt.ylabel("median_house_value")
    plt.xlabel("total_rooms")
    # plot a scatter plot from our data sample
    plt.scatter(sample["total_rooms"], sample["median_house_value"])
    # display graph
    plt.show()

    # those initial sanity checks above suggest that our line is way off and we should be able to find a much better one
    # lets tweak the model parameters - using train_model function
    # train_model(learning_rate=0.00001, steps=100, batch_size=1, my_dataframe=california_housing_dataframe)
    train_model(learning_rate=0.005, steps=500, batch_size=10, my_dataframe=california_housing_dataframe)


if __name__ == '__main__':
    main()
