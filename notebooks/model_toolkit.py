import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
from sklearn.preprocessing import KBinsDiscretizer
import pickle
import tensorflow as tf
import pandas as pd


def draw_square(origin, size, facecolor, edgecolor):
    """
    Draw a square on the given axes.
    :param origin: Tuple (x, y) representing the bottom-left corner of the square
    :param size: Length of the square's side
    :param facecolor: Fill color of the square
    :param edgecolor: Border color of the square
    """
    square = patches.Rectangle(origin, size, size, linewidth=1, edgecolor=edgecolor, facecolor=facecolor)
    plt.gca().add_patch(square)


def calculate_confusion_matrix(y_predicted, y_true):
    vector = [1, 1 / 3, 1 / 3 * 1 / 4]
    coordinates = []
    for i in range(0, len(y_predicted)):
        coordinates.append((np.dot(y_predicted[i], vector), np.dot(y_true[i], vector)))
    return np.array(coordinates)


def plot_confusion_matrix(confusion_matrix):
    X, Y = confusion_matrix.T
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    for x in range(0, 6):
        draw_square((x, x), 1, edgecolor='red', facecolor='white')
        for i in range(0, 9):
            draw_square((x + 1 / 3 * (i % 3), x + 1 / 3 * (i // 3)), 1 / 3, facecolor='white', edgecolor='red')
    plt.plot([0, 5], [0, 5])
    plt.hexbin(X, Y, gridsize=(50, 50), cmap='gray_r', bins='log')
    plt.colorbar(label='Log Density')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return plt


def load_dataset(*paths):
    files = []
    for path in paths:
        files.extend(glob.glob(path, recursive=True))
    instances = [np.load(result) for result in files]
    array = np.array([instance[:, :92][1] for instance in instances])
    df = pd.DataFrame(array)
    return df


def discretize(space, labels):
    transformed_X = np.array(space).reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=len(space), encode='ordinal', strategy='uniform', subsample=None)
    est.fit(transformed_X)
    return est.transform(np.array(labels).reshape(-1, 1)).reshape(-1)


def log_results(occultation_features, *datasets):
    discretizer = Discretizer()
    df = load_dataset(*datasets)
    occultation_features = discretizer.discretize_occultation_features(occultation_features)
    model = tf.keras.models.load_model('checkpoints/model_classification.keras')
    predictions = model.predict(df, verbose=False)
    predictions = np.array([np.argmax(prediction, axis=1) for prediction in predictions]).T
    y_true = np.array([occultation_features['diameter'].to_numpy(), occultation_features['distance'].to_numpy(),
                       occultation_features['impact_parameter'].to_numpy()]).T
    confusion_matrix = calculate_confusion_matrix(predictions, y_true)
    plt = plot_confusion_matrix(confusion_matrix)
    return plt
