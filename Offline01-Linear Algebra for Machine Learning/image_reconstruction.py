"""
Singular Value Decomposition (SVD) is a powerful technique in linear algebra that decomposes a matrix into three simpler matrices.
Used heavily in the fields of data science and machine learning for dimensionality reduction, data compression, and data analysis.
SVD is a generalization of eigen decomposition for non-square matrices.

A=UΣV^T
A is an m×n matrix
U is an m×m orthogonal matrix
Σ is an m×n diagonal matrix
V^T is an n×n orthogonal matrix

The diagonal entries of Σ are known as the singular values of A.
The columns of U and V are known as the left-singular vectors and right-singular vectors of A, respectively.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(file_path, resize_dim=500):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (resize_dim, resize_dim))
    return image

def low_rank_approximation(A, k):
    # Perform SVD
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncate to k singular values
    U_k = U[:, :k]                  #first k columns of the matrix U. These columns correspond to the left singular vectors
    Sigma_k = np.diag(Sigma[:k])    #taking the first k singular values from Sigma
    Vt_k = Vt[:k, :]                #first k rows of the matrix V^T. These rows correspond to the right singular vectors

    # Reconstruct the lowe ranked matrix
    A_k = np.dot(U_k, np.dot(Sigma_k, Vt_k))

    return A_k


def plot_images(images, titles):
    num_plots = len(images)

    # Determine the number of rows and columns for subplots
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)

    for ax, (img, title) in zip(axes.flatten(), zip(images, titles)):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()



if __name__ == "__main__":
    file_path = "E:\L4-T2\CSE-472-Machine-Learning-Sessional\Offline01-Linear Algebra for Machine Learning\image.jpg"
    image = load_and_preprocess_image(file_path)

    # Perform SVD and reconstruct images for different values of k
    k_values = [1, 5, 10, 20, 30, 40, 45, 50, 100, 200, 400, 900]
    reconstructed_images = []

    for k in k_values:
        A_k = low_rank_approximation(image, k)
        reconstructed_images.append(A_k)

    # Plot the original and reconstructed images for different k values
    plot_images(reconstructed_images, [f"Rank={k}" for k in k_values])