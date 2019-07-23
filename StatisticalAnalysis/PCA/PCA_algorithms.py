'''Methods for Principal Component Analysis'''

# Author: Erick Armingol

class PCAResults:
    '''Class Object that generates the PCA results and all the parameters about variables and individuals contributions

    Parameters
    ----------

    X : array-like
        The entire data set used. Each row is a sample and each column is a feature of the samples.

    n_components : int, None by default
        First n-components to report.

    Attributes
    ----------
    X : array-like
        The entire data set used. Each row is a sample and each column is a feature of the samples.
        
    pca : class PCA
        Class object from sklearn to perform Principal Component Analysis.

    explained_var : array
        Explained variance by each feature.

    loadings : array
        The correlation between a variable and a PC is called loading. The loadings are the coordinates of each feature
        given the coordinates analyzed with PCA. They represet correlation of each feature with each principal component.

    var_cos2 :  array
        Cos-squared value of each variable for each component. Cos2 = loadings^2.

    var_contrib : array
        Contribution percentage of each variable to each component.

    ind_coords : array
        Coordinates for each sample or individuals after PCA transformation.

    ind_contrib : array
        Contribution percentage of each sample or individuals to each component.
    '''

    def __init__(self, X, n_components = None):
        if n_components == None:
            n_components = len(X[1,:])
        self.X = X
        self.perform_PCA(self.X, n_components)

    def perform_PCA(self, X, n_components):
        '''
        Parameters
        ----------
        X : array-like
            The entire data set used. Each row is a sample and each column is a feature of the sample.

        n_components : int
            First n-components to report.
        '''

        import numpy as np

        from sklearn import decomposition
        from sklearn.preprocessing import StandardScaler

        # Normalize data
        X_std = StandardScaler().fit_transform(X)

        # Perform PCA
        pca = decomposition.PCA(n_components = n_components)
        pca.fit(X_std)

        # Additional information
        explained_var = pca.explained_variance_ratio_   # Percentage of variance explanation for each component
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_) # Correlation of each variable to components
        ind_coords = pca.transform(X_std)   # Coordinates of each sample after PCA transformation
        var_cos2 = loadings ** 2
        ind_coords2 = ind_coords ** 2

        var_contrib = var_cos2
        for i in range(var_cos2.shape[1]):
            divisor = sum(var_cos2[:,i])
            var_contrib[:, i] = 100 * var_cos2[:, i] / divisor

        ind_contrib = ind_coords2
        for i in range(ind_coords2.shape[1]):
            divisor = sum(ind_coords2[:,i])
            ind_contrib[:, i] = 100 * ind_coords2[:, i] / divisor

        self.pca = pca
        self.explained_var = explained_var
        self.loadings = loadings
        self.var_cos2 = var_cos2
        self.var_contrib = var_contrib
        self.ind_coords = ind_coords
        self.ind_contrib = ind_contrib
    
    def plots(self, feature_names, PC1=1, PC2=2, filename=None):
        '''This function plots loadings and rotated coordinates for samples in the dataset
        
        Parameters
        ----------
        feature_names : array-like
            List of names for the features used in the original dataset.
            
        PC1 : int, 1 by default
            Principal component to plot in x-axis.
            
        PC2 : int, 2 by default
            Principal component to plot in y-axis.
            
        filename : string, None by default
            Name to save the plot.
            
        Returns
        -------
        fig : matplotlib.pyplot.figure
            A matplotlib figure object.

        ax : matplotlib.axes
            A matplotlib axes object.
        '''
        import numpy as np
        fig, ax = PlotLoadings(self.loadings[:,PC1-1],
                               self.loadings[:,PC2-1],
                               'PC{} ({}%)'.format(PC1, round(self.explained_var[PC1-1] * 100, 2)),
                               'PC{} ({}%)'.format(PC2, round(self.explained_var[PC2-1] * 100, 2)),
                               feature_names)
        
        ax.scatter(self.ind_coords[:,PC1-1], self.ind_coords[:,PC2-1], color='gray', s=5)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        
        max_val = np.nanmax(np.concatenate([self.ind_coords[:,PC1-1], self.ind_coords[:,PC2-1]]))
        max_val = np.nanmax(list(x_lim) + list(y_lim) + [max_val])
        ax.set_xlim((-1*max_val, max_val))
        ax.set_ylim((-1*max_val, max_val))
        return fig, ax
               

def PlotLoadings(x, y, x_label, y_label, feature_names, dpi=300, filename = None):
    '''This function plots the loadings as arrows and show their respective names.

    Parameters
    ----------
    x : array
        Coordinates of X-axis.

    y : array
        Coordinates of Y-axis

    x_label : string
        X-axis label.

    y_label : string
        Y-axis label.

    feature_names : array-like
        List of names for the features used in the original dataset.
        
    dpi : int, 300 by default
        Value for dpi to save figure.

    filename : string, None by default
        Name to save the plot.
        
    Returns
    -------
    fig : matplotlib.pyplot.figure
        A matplotlib figure object.
        
    ax : matplotlib.axes
        A matplotlib axes object.
    '''
    import matplotlib.pyplot as plt
    xvector = x
    yvector = y

    fig = plt.figure(figsize=(2000 / dpi, 2000 / dpi), dpi=dpi)
    ax = fig.gca()
    ax.cla()

    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    for i in range(len(xvector)):
        ax.arrow(0, 0, xvector[i], yvector[i],
                 color='r', width=0.0005, head_width=0.025)
        ax.text(xvector[i] * 1.1, yvector[i] * 1.1,
                list(feature_names)[i], color='k', fontdict={'family': 'serif', 'size': 14, }, wrap=True)
    circle = plt.Circle((0, 0), 1.0, color='k', linestyle='--', fill=False)
    ax.add_artist(circle)
    if filename is not None:
        plt.savefig(filename, dpi=dpi)
    return fig, ax
