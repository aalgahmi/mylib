import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

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
    M = X.shape[1]
    columns = [ f"x{i + 1}" for i in range(M) ] if features is None else features.copy()
        
    if y is not None:
        if y.ndim == 1:
            y = y.reshape(len(X), -1)
            
        T = y.shape[1] # number of target columns
        if T == 1:
            columns.append("y" if target is None else target)
        else:
            columns += [ f"y{i + 1}" for i in range(T) ] if target is None else target.copy()
     
    return pd.DataFrame(np.concatenate([X, y], axis=1), columns=columns)
    
def from_dataframe(df, ntargets=1):
    """
    Separates a given data frame into Numpy input and target arrays.
    - df: The data frame 
    - ntargets: The number of target columns at the end of the data frame.
    """
    ntargets = 0 if ntargets is None else ntargets
    if ntargets > 0:
        X = df.iloc[:, :-ntargets].values.squeeze()
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
    if shuffle is True:
        rgen = np.random.RandomState(random_state)
        indexes = rgen.permutation(len(X))
        X, y = X[indexes], y[indexes]

    if not isinstance(test_size, float) or test_size < 0.0 or test_size > 1.0:
        raise TypeError("Only fractions between ]0,1[ are allowed for test_size.")

    split_ndx = int(test_size * len(X))
    
    if y.ndim == 1:
        return X[split_ndx:, :], X[0:split_ndx, :], y[split_ndx:], y[0:split_ndx]
    else:
        return X[split_ndx:, :], X[0:split_ndx, :], y[split_ndx:, :], y[0:split_ndx, :]


#############################################################################################
#
# Model performance functions
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
    
#############################################################################################
#
# Plotting decision regions
#
#############################################################################################
def plot_decision_regions(X, y, learner, resolution=0.1, title="Decision regions", ax=None, figsize=(16,8)):
    if X.shape[1] != 2:
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


class DecisionTreeBase:
    def __init__(self, max_depth=None, typ='cls'):
        self.max_depth = max_depth
        self.typ = typ
        
    def fit(self, X, y, indices = None):
        self.X = X
        self.y = y
        self.ds = to_dataframe(X, y)
        
        if indices is None: 
            indices=np.arange(len(self.y))
            
        self.indices = indices
        self.root = self.Node(self, self.indices, 0)
        
        return self
    
    def output_value(self, y):
        if self.typ == 'cls':
            return st.mode(y)[0][0]
        else:
            return np.mean(y)
    
    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return (-p * np.log2(p)).sum()

    def info_gain(self, df, feature):
        p = df[feature].value_counts() / len(df)

        for v in p.index:
            p.loc[v] *= self.entropy(df[df[feature] == v])

        return self.entropy(df) - p.sum()

    
    def impurity_measure(self, y):
        if self.typ == 'cls':
            return self.entropy(y)
        else:
            return np.var(y)
    
    def best_feature_split(self, node, feature_index):
        """
        When called on all the features, it finds the best feature to 
        split by and at what threshold. 
        
        NOTE that: lhs = left hand side and rhs = right hand side
        """
        X, y = self.X[node.indices,feature_index], self.y[node.indices]
        var_d = self.impurity_measure(y)
        d = len(y)
        
        """Let us sort the feature values"""
        sort_idxs = np.argsort(X)
        sorted_X, sorted_y = X[sort_idxs], y[sort_idxs]

        for i in range(0, node.N - 1):
            xi,yi = sorted_X[i],sorted_y[i]
            
            if i < 2 or xi == sorted_X[i + 1]:
                continue
            
            d_lhs = len(y[np.nonzero(X <= xi )])
            d_rhs = len(y[np.nonzero(X > xi )])
            
            var_lhs = self.impurity_measure(y[np.nonzero(X <= xi )]) #np.var(y[np.nonzero(X <= xi )])
            var_rhs = self.impurity_measure(y[np.nonzero(X > xi )])  #np.var(y[np.nonzero(X > xi )])
            var_reduction = var_d - ((d_lhs / d) * var_lhs + (d_rhs / d) * var_rhs)            
                            
            if var_reduction > node.split_var_reduction: 
                node.split_feature_index = feature_index
                node.split_var_reduction = var_reduction
                node.split_threshold = xi
                
    def print_tree(self, node=None):
        if node is None: node = self.root
        print(repr(node))
        if node.left: self.print_tree(node.left)
        if node.right: self.print_tree(node.right)


    def predict(self, unseen, node=None):
        if unseen.ndim == 1:
            if node is None:
                node = self.root
                
            if node.is_leaf: return node.value
            node = node.left if unseen[node.split_feature_index] <= node.split_threshold else node.right
            return self.predict(unseen, node)
        else:
            if isinstance(unseen, pd.DataFrame):
                unseen = unseen.values
                
            return np.array([self.predict(unseen[i,:]) for i in range(len(unseen))]) 
        
    class Node:
        """
        An internal class representing the nodes of the tree.
        """
        def __init__(self, tree, indices, depth):
            self.tree, self.indices, self.depth = tree, indices, depth            
            self.N, self.M = len(self.indices), self.tree.X.shape[1]
            self.value = self.tree.output_value(self.tree.y[self.indices])
            self.split_feature_index = 0
            self.split_var_reduction = 0
            self.left, self.right = None, None
            
            if tree.max_depth is not None and tree.max_depth == depth:
                return
            
            """
            Find the best split across all features(dimentions) based on the minimum
            variance reduction.
            """
            for feature_index in range(self.M): 
                self.tree.best_feature_split(self, feature_index)
                
            if self.split_var_reduction == 0: 
                return

            feature = self.tree.X[self.indices, self.split_feature_index]

            """Binary branching based on the best found feature and threshold."""
            lhs_indices = np.nonzero(feature <= self.split_threshold)[0]
            rhs_indices = np.nonzero(feature > self.split_threshold)[0]
            
            self.left = self.tree.Node(self.tree, self.indices[lhs_indices], self.depth + 1)
            self.right= self.tree.Node(self.tree, self.indices[rhs_indices], self.depth + 1)

        @property
        def split_name(self): 
            return self.tree.ds.columns[self.split_feature_index]

        @property
        def is_leaf(self): 
            return self.left is None and self.right is None

        @property
        def is_root(self): 
            return self == self.tree.root
           
        def __repr__(self):
            out = f'|{" | " * self.depth}'
            if not self.is_leaf:
                out += f'{self.split_name} (split: {self.split_threshold}), output value: {self.value}'
            else:
                out += f'Leaf (value: {self.value})'
                
            return out

class DecisionTreeClassifier2 (DecisionTreeBase):
    """
    Used for classification when features are continuous.
    """
    def __init__(self, max_depth=None):
        super().__init__(max_depth, typ='cls')
    
class DecisionTreeRegressor (DecisionTreeBase):
    def __init__(self, max_depth=None):
        super().__init__(max_depth, typ='reg')