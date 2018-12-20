import graphviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

def display_tree(dt,feature_names,class_names,out_file=None):
    dot_data = tree.export_graphviz(dt, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("Outputs/"+out_file)

    dot_data = tree.export_graphviz(dt, out_file=None, 
                      feature_names=feature_names,  
                      class_names=class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
    graph = graphviz.Source(dot_data)  
    return graph 


def display_classification_boundary(clf, pairidx, title, iris, X, y,xlabel='',ylabel=''):
    plot_step = 0.02
    n_classes = 3
    plot_colors = "ryb"
    
    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    if xlabel:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1])
    plt.title(title)
    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)