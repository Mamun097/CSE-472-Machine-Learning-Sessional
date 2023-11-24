import numpy as np

"""
The np.triu function is used to extract the upper triangular part, 
and np.triu(upper_triangular, k=1).T generates the lower triangular part 
    by taking the transpose of the strictly upper triangular part.
Adding these two components together creates a symmetric matrix.
"""
def symmetric_invertible_matrix(n):
    while True:
        random_matrix = np.random.randint(-10, 10, size=(n, n))
        A = np.triu(random_matrix) + np.triu(random_matrix, 1).T
        if np.linalg.det(A) != 0:
            break
    return A

def eigen_decomposition(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors

def reconstruct_matrix(eigenvalues, eigenvectors):
    A_reconstructed = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.linalg.inv(eigenvectors)))
    return A_reconstructed

def check_reconstruction(A, A_reconstructed):
    reconstruction_success = np.allclose(A, A_reconstructed)
    return reconstruction_success
"""
np.allclose(a, b) is a NumPy function that checks if all elements of arrays a and b are approximately equal within some tolerance.
This function is useful for comparing two arrays when we expect them to be almost equal 
due to the limitations of floating-point arithmetic.
"""


def main():
    n = int(input("Enter the dimensions of the matrix (n): "))
    A = symmetric_invertible_matrix(n)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = eigen_decomposition(A)
    
    # Reconstruct the matrix
    A_reconstructed = reconstruct_matrix(eigenvalues, eigenvectors)
    
    # Check if the reconstruction worked properly
    reconstruction_success = check_reconstruction(A, A_reconstructed)
    
    # Print the results
    print("\nOriginal Matrix A:")
    print(A)
    
    print("\nReconstructed Matrix A:")
    print(A_reconstructed)
    
    if reconstruction_success:
        print("\nReconstruction successful!")
    else:
        print("\nReconstruction failed!")


if __name__ == '__main__':
    main()
