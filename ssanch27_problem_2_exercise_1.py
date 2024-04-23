# Import modules
import matplotlib.pyplot as plt
import numpy as np

# Define fuction of the problem
def run_problem_2(np_file):
    """
    In this problem I will approach the creation of summary images from all frames,
    both with proposed statistics and for some I thought of

    This problem requires for the .tiff file to be passed as numpy (np_file) and returns
    the summary images created only wiht the proposed statistics (mean_frame, median_frame
    and variance_frame)
    """

    # Part A
    print("PART A")

    # Get summary images
    mean_frame = np.mean(np_file, axis=0)
    median_frame = np.median(np_file, axis=0)
    variance_frame = np.var(np_file, axis=0)

    # Plot summary images
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(mean_frame, cmap="gray")
    plt.axis("off")
    plt.title(f"Mean frame")
    plt.subplot(1,3,2)
    plt.imshow(median_frame, cmap="gray")
    plt.axis("off")
    plt.title(f"Median frame")
    plt.subplot(1,3,3)
    plt.imshow(variance_frame, cmap="gray")
    plt.axis("off")
    plt.title(f"Variance frame")
    plt.tight_layout()
    plt.show()

    # Part B
    print("Part B")

    # Print proposed statistics
    print("The 2 proposed statistics are Standard Deviation and Max Value Projection")

    # Get summary images
    std_frame = np.std(np_file, axis=0)
    max_frame = np.max(np_file, axis=0)

    # Plot summary images
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(std_frame, cmap="gray")
    plt.axis("off")
    plt.title(f"Standard Deviation frame")
    plt.subplot(1,2,2)
    plt.imshow(max_frame, cmap="gray")
    plt.axis("off")
    plt.title(f"Max Projection frame")
    plt.tight_layout()
    plt.show()

    return mean_frame, median_frame, variance_frame
