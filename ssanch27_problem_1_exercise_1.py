# Import modules
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from scipy.signal import fftconvolve

# Define fuction of the problem
def run_problem_1(np_file, COLAB):
    """
    In this problem I will approach the creation of an animation for different frames
    as well as seeing correlation maps for different offsets

    This problem requires for the .tiff file to be passed as numpy (np_file) and doesn't
    return anything (None). It also gets a variable that controls if the file is being
    executed on Colab or not (COLAB), because the animation doesn't work on Colab
    """

    # Part A
    print("PART A")

    # Get basic info of the file
    print(f"Type: {type(np_file)}")
    print(f"Shape: {np_file.shape}")
    print(f"Range of values: [{np.min(np_file)} - {np.max(np_file)}]")

    # Plot example of one frame
    plt.figure()
    plt.imshow(np_file[0], cmap="gray")
    plt.axis("off")
    plt.title("Example of one frame")
    plt.show()

    # Do animation of the file
    if not COLAB:
        fig = px.imshow(np_file, animation_frame=0, binary_string=True, 
                        labels=dict(animation_frame="Frame"))
        fig.update_layout(width=500, height=500)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()
    else:
        print("The animation doesn't show properly in Colab. If you run the file locally the animation will appear.\nIn the repo there is available one screenshot showing that the animation works locally.")

    print()

    # Part B
    print("PART B")

    # Defined frames that are offset
    frame_1 = 0
    frame_2 = 400

    # Get correlation
    correlation_frame = fftconvolve(np_file[frame_1], np_file[frame_2], mode = "same")

    # Plot frames and correlation
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(np_file[frame_1], cmap="gray")
    plt.axis("off")
    plt.title(f"Frame {frame_1 + 1}")
    plt.subplot(1,3,2)
    plt.imshow(np_file[frame_2], cmap="gray")
    plt.axis("off")
    plt.title(f"Frame {frame_2 + 1}")
    plt.subplot(1,3,3)
    plt.imshow(correlation_frame, cmap="gray")
    plt.axis("off")
    plt.title(f"Correlation frame")
    plt.tight_layout()
    plt.show()

    # Set different offsets
    offsets = [100, 200, 300, 400, 499]
    max_correlation_idxs = []
    correlation_frames = []

    # Get correlation and max corr idx by offset
    for offset in offsets:
        correlation_frame = fftconvolve(np_file[frame_1], np_file[frame_1 + offset], mode = "same")
        correlation_frames.append(correlation_frame)
        max_correlation_idxs.append(np.unravel_index(np.argmax(correlation_frame), correlation_frame.shape))

    # Plot different correlation maps and correlation
    plt.figure()
    for i in range(len(offsets)):
        plt.subplot(1,5,i + 1)
        plt.imshow(correlation_frames[i], cmap="gray")
        plt.axis("off")
        plt.title(f"Correlation\nwith frame {frame_1}\nand {frame_1 + offsets[i]}")
    plt.tight_layout()
    plt.show()

    # Print location of max correlation
    for i in range(len(offsets)):
        print(f"Region with max correlation for offset {offsets[i]} around ({max_correlation_idxs[i][0]},{max_correlation_idxs[i][0]})") 

    return None
