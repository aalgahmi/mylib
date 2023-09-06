import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.utils import check_array

#############################################################################################
#
# Dataset functions
#
#############################################################################################
def to_dataframe(X, y=None, features=None, target=None):
    """
    Puts X and y into a data frame and prints it.
    - X, y: The input and target data.
    - features: The names of the input data features.
    - target: The name of the target column.
    """
    check_array(X, ensure_2d=True, dtype=None)
    M = X.shape[1]
    columns = [ f"x{i + 1}" for i in range(M) ] if features is None else features.copy()
        
    if y is not None:
        if y.ndim == 1:
            y = y.reshape(len(X), -1)
            
        T = 1 if y.ndim == 1 else y.shape[1] # number of target columns
        if T == 1:
            columns.append("y" if target is None else target)
        else:
            columns += [ f"y{i + 1}" for i in range(T) ] if target is None else target.copy()
     
    return pd.DataFrame(
        np.concatenate([X, y if y.ndim > 1 else y.reshape(-1,1)], axis=1), columns=columns)
    
def from_dataframe(df, ntargets=1):
    """
    Separates a given data frame into Numpy input and target arrays.
    - df: The data frame 
    - ntargets: The number of target columns at the end of the data frame.
    """
    ntargets = 0 if ntargets is None else ntargets
    if ntargets > 0:
        X = df.iloc[:, :-ntargets].values
        y = df.iloc[:, -ntargets:].values.squeeze()
        features = list(df.columns[:-ntargets])
        
        targets = list(df.columns[-ntargets:])
        
        return X, y, features, targets[0] if len(targets) == 1 else targets
    else:
        X = df.values
        features = list(df.columns)
        
        return X, features

def print_dataset(X, y=None, name=None, features=None, target=None):
    """
    Puts X and y into a data frame and prints it.
    - X, y: The input and target data.
    - features: The names of the input data features.
    - target: The name of the target column.
    """
    if name is not None:
        print(name)
        
    print(to_dataframe(X, y=y, features=features, target=target))
    
def shuffled(X, y=None, random_state=None):
    """
    Shuffles the X, y.
    """
    rgen = np.random.RandomState(random_state)
    
    indexes = rgen.permutation(len(X))

    return X[indexes], y[indexes]

def train_test_split(X, y, test_size=.25, shuffle=True, random_state=None):
    """
    Splits the dataset into a training set and a test set. If test_portion 
    is specified, return that portion of the dataset as test and the rest 
    as training.
    """
    check_array(X, ensure_2d=True, dtype=None)
    
    if shuffle is True:
        rgen = np.random.RandomState(random_state)
        indexes = rgen.permutation(len(X))
        X, y = X[indexes], y[indexes]

    if not isinstance(test_size, float) or test_size < 0.0 or test_size > 1.0:
        raise TypeError("Only fractions between ]0,1[ are allowed for test_size.")

    split_ndx = int(test_size * len(X))
    
    return X[split_ndx:,], X[0:split_ndx,], y[split_ndx:,], y[0:split_ndx,]


#############################################################################################
#
# Model selection/performance functions
#
#############################################################################################
def confusion_matrix(targets, predicted):
    """
    A confusion matrix C is such that C[i, j] is equal to the number 
    of observations known to be in i and predicted to be in group j.
    """
    # Convert classes into index
    targets = targets.flatten()
    predicted = predicted.flatten()
    labels = np.unique(np.concatenate([targets, predicted], axis=0))
    np.sort(labels)
    nLabels = len(labels)
    
    label_to_index = { lbl: ndx for ndx, lbl in enumerate(labels) } 
    
    y_true = np.array([label_to_index.get(lbl) for lbl in targets])
    y_pred = np.array([label_to_index.get(lbl) for lbl in predicted])
    
    cm = np.full((nLabels, nLabels), 0)
    for i in range(nLabels):
        for j in range(nLabels):
            cm[i,j] = np.sum(np.where(y_true == i, 1, 0) * np.where(y_pred==j, 1, 0))
    
    return cm

def confusion_matrix_accuracy(cm):
    return np.trace(cm) / np.sum(cm)

def plot_confusion_matrix_with_accuracy(classifier, X_test, y_test, ax=None):
    actual, predicted = y_test, classifier.predict(X_test)
    cm = confusion_matrix(actual, predicted)
    ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax)
    print(f"Accuracy: {accuracy_score(actual, predicted)}")
    
def plot_cv_curve(x, y, xlabel, ylabel, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16,5))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    
#############################################################################################
#
# Plotting decision regions
#
#############################################################################################
def plot_decision_regions(X, y, learner, resolution=0.1, title="Decision regions", ax=None, figsize=(16,8)):
    if X.ndim != 2:
        raise TypeError("Input data must be two-dimensional. That is the first X param can only have two features.")

    if y.ndim == 1:
        y = y.reshape(len(X), -1)
        
    D = np.concatenate([X, y], axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    # Create color maps
    cmap_light = ListedColormap(['#79dfc1', '#feb272', '#a370f7', '#6ea8fe', '#dee2e6', '#ea868f'])
    cmap_bold = ListedColormap(['#20c997', '#fd7e14', '#6610f2', '#0d6efd', '#adb5bd', '#dc3545'])
    
    # Plot the decision boundary.
    x_min, x_max = D[:,0].min() - 1, D[:,0].max() + 1
    y_min, y_max = D[:,1].min() - 1, D[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = learner.predict(np.array([xx.ravel(), yy.ravel()]).T)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.rcParams['pcolor.shading'] ='nearest'
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the points
    ax.scatter(
        x=D[:, 0],
        y=D[:, 1], c=D[:,-1], cmap=cmap_bold,
        edgecolor='k', s=60)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plt.legend(loc='best')
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(title)

    # plt.show()
