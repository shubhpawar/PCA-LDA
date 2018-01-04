"""
@author: Shubham Shantaram Pawar
"""

# importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plotting raw data
def plotOriginalData(X, y):
    V1_ = np.where(y==1)[0]
    V2_ = np.where(y==0)[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot of Raw Data')
    ax.scatter(X[V1_][:,0], X[V1_][:,1], color='blue', label='class 1', marker='o')
    ax.scatter(X[V2_][:,0], X[V2_][:,1], color='red', label='class 0', marker='+')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.legend()
    fig.set_size_inches(10, 6)
    plt.savefig('Scatter Plot of Raw Data')
    fig.show()
   
# plotting raw data with PC1 axis
def plotPC1Axis(X, y, pca_results):
    V1_ = np.where(y==1)[0]
    V2_ = np.where(y==0)[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Raw Data with PC1 Axis')
    ax.scatter(X[V1_][:,0], X[V1_][:,1], color='blue', label='class 1', marker='o')
    ax.scatter(X[V2_][:,0], X[V2_][:,1], color='red', label='class 0', marker='+')
    plt.plot([0, -50*pca_results['eig_pairs'][0][1][0]],
             [0, -50*pca_results['eig_pairs'][0][1][1]], 'g-', label = 'PC1 Axis')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.legend()
    fig.set_size_inches(10, 6)
    plt.savefig('Raw Data with PC1 Axis')
    fig.show()

# plotting projection after LDA
def plotLDAProjection(projection):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Projection of raw data onto W')
    ax.set_xlabel('projection')
    ax.set_ylabel('')
    ax.plot(projection[:30], np.zeros(30), linestyle='None', marker='o', markersize=7, color='blue', label='class 1')
    ax.plot(projection[30:], np.zeros(30), linestyle='None', marker='o', markersize=7, color='red', label='class 0')
    ax.legend()
    fig.set_size_inches(10, 6)
    plt.savefig('Projection of raw data onto W')
    fig.show()

# plotting W axis with raw data and PC1 axis
def plotWAxis(X, y, W, pca_results):
    V1_ = np.where(y==1)[0]
    V2_ = np.where(y==0)[0]
    
    W_scaled = W * 12.0 / W[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Raw Data with PC1 Axis and W Axis')
    ax.scatter(X[V1_][:,0], X[V1_][:,1], color='blue', label='class 1', marker='o')
    ax.scatter(X[V2_][:,0], X[V2_][:,1], color='red', label='class 0', marker='+')
    plt.plot([0, -50*pca_results['eig_pairs'][0][1][0]],
             [0, -50*pca_results['eig_pairs'][0][1][1]], 'g-', label = 'PC1 Axis')
    ax.plot([0, W_scaled[0]], [0, W_scaled[1]], color='orange', label = 'W Axis')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.legend()
    fig.set_size_inches(10, 6)
    plt.savefig('Raw Data with PC1 Axis and W Axis')
    fig.show()
    
# plotting projection after PCA
def plotPC1Projection(projection):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Projection of raw data onto PC1')
    ax.set_xlabel('projection')
    ax.set_ylabel('')
    ax.plot(projection[:30], np.zeros(30), linestyle='None', marker='o', markersize=7, color='blue', label='class 1')
    ax.plot(projection[30:], np.zeros(30), linestyle='None', marker='o', markersize=7, color='red', label='class 0')
    ax.legend()
    fig.set_size_inches(10, 6)
    plt.savefig('Projection of raw data onto PC1')
    fig.show()
    
# function to compute feature-wise mean
def computeMean(x):
    return np.mean(x, axis = 0)

# function to compute feature-wise mean for each class
def computeMeanVectors(X,y):
    mean_vectors = []
    for class_label in np.unique(y):
        mean_vectors.append(computeMean(X[y == class_label]))
    
    return mean_vectors
    
# function to compute Scatter-Within
def calculateSWithin(X, y):
    n_dim = X.shape[1]
    S_Within = np.zeros([n_dim, n_dim])
    mean_vectors = computeMeanVectors(X, y)
    
    for class_label in np.unique(y):
        within_scatter = np.zeros([n_dim, n_dim])
        
        for sample in X[y == class_label]:
            sample, vec = sample.reshape(n_dim, 1), mean_vectors[int(class_label)].reshape(n_dim, 1)
            within_scatter += np.dot(sample - vec, (sample - vec).T)
        S_Within += within_scatter
    
    return S_Within

# function to compute Scatter-Between
def calculateSBetween(X, y):
    n_dim = X.shape[1]
    S_Between = np.zeros([n_dim, n_dim])
    mean_global = computeMean(X)
    mean_vectors = computeMeanVectors(X, y)
    
    for class_label in np.unique(y):
        N = X[y == class_label].shape[0]
        mean_global, vec = mean_global.reshape(n_dim, 1), mean_vectors[int(class_label)].reshape(n_dim, 1)
        S_Between += N * np.dot(vec - mean_global, (vec - mean_global).T)
    
    return S_Between
    
# function to perform Linear Discriminant Analysis (LDA)
def do_LDA(X, y):
    m, n = X.shape
    
    # computing Scatter_Between and Scatter_Within matrices using functions
    S_Between = calculateSBetween(X, y)
    S_Within = calculateSWithin(X, y)
    
    # computing eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(S_Within), S_Between))
    
    # forming eigenvalue-eigenvector pairs
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # sorting eigenvalue-eigenvector pairs in descending order
    eig_pairs.sort(reverse=True, key=(lambda x: x[0]))
    
    # choosing eigenvectors with the largest eigenvalue
    W = eig_pairs[0][1].reshape(n,1)
    
    # transforming the samples onto the new subspace
    projection = np.matmul(X, W)
    
    print('\nThe variance of the projections onto the W axis: ', np.var(projection))
    
    projection = projection.tolist()
    
    return W, projection

# function to do Principal Component Analysis (PCA)
def PCA(X):
    m, n = X.shape
    
    pca_data = X
            
    # computing co-variance matrix
    cov_mat = np.cov(pca_data.T)
            
    # computing eigenvalues and eigenvectors for the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    #sorting in descending order by eigenvalues
    eig_pairs.sort(reverse=True, key=(lambda x: x[0]))
    
    pcaScores = np.matmul(pca_data, eig_vecs)
    
    matrix_w = np.hstack((eig_pairs[0][1].reshape(n,1)))
    
    percentVarianceExplained = 100 * eig_pairs[0][0] / sum(eig_vals)
    print("\nPC1 Variance: " + str(round(percentVarianceExplained, 2)) + '% variance\n')
    
    percentVarianceExplained = 100 * eig_pairs[1][0] / sum(eig_vals)
    print("\nPC2 Variance: " + str(round(percentVarianceExplained, 2)) + '% variance\n')
    
    projection1 = pca_data.dot(matrix_w)
    
    projection2 = pca_data.dot(np.hstack((eig_pairs[1][1].reshape(n,1))))
    
    print('\nVariance of the projection onto PC1:', np.var(projection1))
    print('\nVariance of the projection onto PC2:', np.var(projection2))
    
    print('\nThe eigenvalues of the covariance matrix used for computing PC1 and PC2 axes:')
    print(eig_pairs[0][0])
    print(eig_pairs[1][0])
    
    # projection
    Y = projection1
    
    pca_results = {'data': X,
                   'eig_pairs': eig_pairs,
                   'scores': pcaScores,
                   'results': Y}
    
    return pca_results

def main():
    df = pd.read_csv(filepath_or_buffer='dataset_1.csv', header=0, sep=',')
    
    df.dropna(how="all", inplace=True)
    df.tail()
            
    data_in = df.values
        
    # input data
    X = data_in[:, [0, 1]]
    # class labels
    y = df.values[:, 2]
    
    #plotting raw data
    plotOriginalData(np.array(X), np.array(y).reshape(len(X), 1))
    
    pca_results = PCA(np.array(X))
    
    # plotting projection after PCA
    plotPC1Projection(pca_results['results'])
    
    # plotting PC1 axis
    plotPC1Axis(np.array(X), np.array(y).reshape(len(X), 1), pca_results)
            
    W, projection = do_LDA(np.matrix(X), np.array(y))
    
    print('\nW from LDA:', W)
    
    # plotting projection after LDA
    plotLDAProjection(projection)
    
    # plotting W axis
    plotWAxis(np.array(X), np.array(y).reshape(len(X), 1), W, pca_results)
    
'''
1. PCA is a technique that finds the direction of maximum variance.
2. LDA tries to find a feature subspace that maximizes class separability.
3. LDA is supervised whereas PCA is unsupervised (ignores class labels).
'''

if __name__ == '__main__':
    main()