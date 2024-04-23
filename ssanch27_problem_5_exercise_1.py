# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF, FastICA

# Define fuction of the problem
def run_problem_5(np_file):
    """
    In this problem I will approach the matrix factorization of the file to avoid
    summary frames. The approach will include variations with PCA, NMF and ICA

    This problem requires for the .tiff file to be passed as numpy (np_file) and doesn't
    return anything (None)
    """

    # Get pixels by time by vectorizing
    vectorized_np_file = np_file.reshape(np_file.shape[0], -1).T

    # Print info
    print(f"Shape of vectorized matrix: {vectorized_np_file.shape}")
    print()

    # Part A
    print("Part A")

    # Define list of #components
    n_components_pca_list = [2, 4, 8]

    # For #components, perform PCA and see results
    for n_components_pca in n_components_pca_list:
        # Print info
        print(f"Beginning iteration of PCA with {n_components_pca} principal components")

        # Get PCA
        this_pca = PCA(n_components=n_components_pca)
        after_pca = this_pca.fit_transform(vectorized_np_file.T).T

        # Print info
        print(f"Shape of output of this iteration of PCA: {after_pca.shape}")

        # Plot PCA values
        plt.figure()
        for i in reversed(range(n_components_pca)):
            if i < 2:
                plt.plot(after_pca[i, :], "r")
            else:
                plt.plot(after_pca[i, :], "b")
        plt.title(f"PCA values for {n_components_pca} components\n[red corresponds to the 2 first components]")
        plt.xlabel("Time (frames)")
        plt.ylabel("Component value (AU)")
        plt.show()

        # Plot PCA plane
        if n_components_pca == 2:
            plt.figure()
            plt.scatter(after_pca[0, :], after_pca[1, :])
            plt.title(f"PCA plane for {n_components_pca} components")
            plt.xlabel("PCA 1 (AU)")
            plt.ylabel("PCA 2 (AU)")
            plt.show()
    
    print()

    # Part B
    print("Part B")

    print("This part takes a couple of minutes to execute do the high number of iterations set in the NMF model to assure it converges.\nLocally it takes 1 minute, but in Colab it can take around 5 minutes")

    # Get NMF model
    n_components_nmf = 2
    model = NMF(n_components=n_components_nmf, init="random", random_state=0, max_iter=750)

    # Train model and get the components
    W = model.fit_transform(vectorized_np_file)
    H = model.components_

    # Plot spatial component of NMF
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(W[:, 0].reshape(np_file.shape[1], np_file.shape[2]), cmap="hot")
    plt.axis("off")
    plt.title(f"First NMF spatial component")
    plt.subplot(1,2,2)
    plt.imshow(W[:, 1].reshape(np_file.shape[1], np_file.shape[2]), cmap="hot")
    plt.axis("off")
    plt.title(f"Second NMF spatial component")
    plt.tight_layout()
    plt.show()

    # Plot temporal component of NMF
    plt.figure()
    plt.plot(H[0,:])
    plt.plot(H[1,:])
    plt.title(f"NMF temporal components for {n_components_nmf} components")
    plt.xlabel("Time (frames)")
    plt.ylabel("Component value (AU)")
    plt.legend("NMF 1")
    plt.show()

    # Plot temporal component plane of NMF
    plt.figure()
    plt.scatter(H[0,:], H[1,:])
    plt.title(f"NMF temporal components plane for {n_components_nmf} components")
    plt.xlabel("NMF 1 (AU)")
    plt.ylabel("NMF 2 (AU)")
    plt.show()

    print()

    # Part C
    print("Part C")

    # Get ICA model
    n_components_ica = 2
    ica = FastICA(n_components=n_components_ica, random_state=0)

    # Train model and get mixing matrix
    S = ica.fit_transform(vectorized_np_file)
    A = ica.mixing_

    # Plot output component of ICA
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(S[:, 0].reshape(np_file.shape[1], np_file.shape[2]), cmap="hot")
    plt.axis("off")
    plt.title(f"First ICA output component")
    plt.subplot(1,2,2)
    plt.imshow(S[:, 1].reshape(np_file.shape[1], np_file.shape[2]), cmap="hot")
    plt.axis("off")
    plt.title(f"Second ICA output component")
    plt.tight_layout()
    plt.show()

    # Plot mixing matrix component plane of ICA
    plt.figure()
    plt.scatter(A[:,0], A[:,1])
    plt.title(f"ICA mixing matrix components plane for {n_components_nmf} components")
    plt.xlabel("ICA 1 (AU)")
    plt.ylabel("ICA 2 (AU)")
    plt.show()

    return None
