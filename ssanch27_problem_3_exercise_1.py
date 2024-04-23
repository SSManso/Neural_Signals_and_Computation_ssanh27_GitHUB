# Import modules
import matplotlib.pyplot as plt
import numpy as np

# Define fuction of the problem
def run_problem_3(summary_frame):
    """
    In this problem I will approach the creation of Regions Of Interest (ROIs) from a
    summary image. The seeds are inputed as literal values from previous visual inspection. For
    this, I will create an intermediate function to get the ROIs from a given summary frame and
    seed

    This problem requires for the summary frame array to be passed (summary_frame) and returns
    a list of binary masks of each ROI (list_rois)
    """

    # Part A
    print("PART A")

    # Create intermediate function
    def get_roi(frame, seed, threshold=None, max_length=50):
        """
        This function will output a binary mask of a ROI for a given frame and seed. For this,
        a threshold is created as 3 times the std of the overall frame (this could be changed later,
        in this case it has been seen from previous trials that this method works for the variance
        summary frame). The function also requires the max size of the ROI (region to loof for) (max_length)
        
        This function WON'T check for the validity of the input parameters

        The function requires for the summary frame array to be passed (frame) and for a seed as
        coordinates y,x (seed) and returns a binary mask of the ROI created (roi), which is the same
        shape as frame. The optional values of threshold and max_length have a default value, so
        they are not necessary
        """

        # Create threshold to consider the ROI (if not given)
        if threshold is None:
            threshold = 3 * np.std(frame)
        
        # Create window to look at in the frame
        min_window_y = np.maximum(seed[0] - (max_length // 2), 0)
        max_window_y = np.minimum(seed[0] + (max_length // 2), frame.shape[0])
        min_window_x = np.maximum(seed[1] - (max_length // 2), 0)
        max_window_x = np.minimum(seed[1] + (max_length // 2), frame.shape[1])

        # Define ROI mask, and only compare with threshold inside window
        roi = np.zeros_like(frame, dtype=bool)
        roi[min_window_y:max_window_y, min_window_x:max_window_x] = frame[min_window_y:max_window_y, min_window_x:max_window_x] > threshold

        return roi

    # Define seeds (format (y,x)). Obatined through visual inspection
    seeds = [
        [50, 350],
        [150, 125],
        [475, 175],
        [400, 400],
        [275, 350],
    ]

    # Get roi masks for each seed
    list_rois = []
    for seed in seeds:
        # Get roi mask
        roi_mask = get_roi(summary_frame, seed)

        # Save roi mask
        list_rois.append(roi_mask)

        # Plot roi mask
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(summary_frame, cmap="gray")
        plt.axis("off")
        plt.title(f"Summary frame")
        plt.subplot(1,3,2)
        plt.imshow(roi_mask, cmap="gray")
        plt.axis("off")
        plt.title(f"ROI mask\nwith seed {seed}")
        plt.tight_layout()
        plt.subplot(1,3,3)
        plt.imshow(summary_frame * roi_mask, cmap="gray")
        plt.axis("off")
        plt.title(f"Summary frame\nwith ROI mask\nwith seed {seed}")
        plt.show()

    return list_rois
