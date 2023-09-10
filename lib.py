import requests, zipfile, io
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.utils import check_array

#############################################################################################
#
# Utility functions
#
#############################################################################################
def download_zip_and_open_a_file(zip_url, filename):
    r = requests.get(url)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    return zf.open(filename)

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

#############################################################################################
#
# Decision tree models
#
#############################################################################################
class DecisionTreeClassifier:
    """ 
    Used for classification when input features are discrete.
    """

    def __init__(self, max_depth=-1, nFeatures=0, criterion='gini'):
        self.tree = pd.Series(dtype=object)
        self.__criterion = criterion
        self.nFeatures = nFeatures
        self.max_depth = max_depth

    def fit(self, X, y, features, target):
        self.features = features
        self.dataset = to_dataframe(X, y, features, target)
        self.make_tree(self.dataset, self.tree, depth=1)

        return self

    def entropy(self, df):
        p = df.iloc[:, -1].value_counts() / len(df)
        return (-p * np.log2(p)).sum()

    def info_gain(self, df, feature):
        p = df[feature].value_counts() / len(df)

        for v in p.index:
            p.loc[v] *= self.entropy(df[df[feature] == v])

        return self.entropy(df) - p.sum()

    def gini(self, df):
        p = df.iloc[:, -1].value_counts() / len(df)
        return 1 - (p**2).sum()

    def weighted_gini(self, df, feature):
        p = df[feature].value_counts() / len(df)

        for v in p.index:
            p.loc[v] *= self.gini(df[df[feature] == v])

        return p.sum()

    def best_feature(self, df):
        features = df.columns[:-1].copy().values
        if self.nFeatures != 0:
            f_indexes = np.arange(min(self.nFeatures, len(features)))
            np.random.shuffle(f_indexes)
            features = features[f_indexes]

        info = pd.DataFrame({"feature": features})
        if self.__criterion == 'gini':
            info['gini'] = [self.weighted_gini(df, f) for f in features]
            return info['feature'][info['gini'].argmin()]
        else:
            info['gain'] = [self.info_gain(df, f) for f in features]
            return info['feature'][info['gain'].argmax()]


    def print_tree(self, name='', node=None, depth=1):
        if node is None:
            node = self.tree

        for f in node.index:
            if isinstance(node[f], tuple):
                if f != '-^-':
                    print(' ' * depth, f, ' => ', node[f], sep='')
            else:
                print(' ' * depth, f, ': ', sep='')
                self.print_tree(f, node[f], depth + 1)

    def make_tree(self, df, node, feature=None, depth=1):
        if self.max_depth == 0:
            return

        if feature is None:
            feature = self.best_feature(df)

        node[feature] = pd.Series(dtype=object)

        # Store the plurality vote class at the feature depth
        # under a "hidden" _^_ key just in case we need it for
        # when the unseen example does not lead to a leaf.
        node[feature]['-^-'] = (feature, df.iloc[:, -1].mode()[0])

        fvalues = df[feature].unique()
        for v in fvalues:
            d = df[df[feature] == v]

            default_class = d.iloc[:, -1].mode()[0]
            if len(d) == 0 or (self.max_depth > 0 and depth >= self.max_depth):
               node[feature][v] = ('L', default_class)
            else:
                n_classes = len(d.iloc[:, -1].unique())
                if n_classes == 1:
                    node[feature][v] = ('L', d.iloc[:, -1].iloc[0])
                elif n_classes > 1:
                    d = d.drop([feature], axis=1)
                    if len(d.columns) == 1: 
                        node[feature][v] = ('L', d.iloc[:, -1].mode()[0])
                    else:
                        next_best_feature = self.best_feature(d)
                        node[feature][v] = pd.Series(dtype=object)
                        self.make_tree(d, node[feature][v], next_best_feature, depth=depth + 1)
                else:
                    pass

    def predict(self, X_unseen, node=None):
        """
        Returns the most probable label (or class) for each unseen input. The
        unseen needs to be dataseries of the same features (as index) as the 
        training data. It can also be a data frame with the same features as 
        the training data
        """
        if X_unseen.ndim == 1:
            x = pd.Series(X_unseen, index=self.features)
            if node is None:
                node = self.tree

            feature = node.index[0]

            children = node[feature]
            value = x[feature]
            for c in children.index:
                if c == value:
                    if isinstance(children[c], tuple):
                        return children[c][1]
                    else:
                        return self.predict(x, children[c])

            # At this point, a leaf was not reached. So we return
            # a plurality vote at the deepest node reached.
            return children['-^-'][1]
        else:
            return np.array([self.predict(X_unseen[i,:]) for i in range(len(X_unseen))]) 

    def __repr__(self):
        return repr(self.tree)