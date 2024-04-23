# Import modules
import matplotlib.pyplot as plt
import numpy as np

# Define fuction of the problem
def run_problem_4(np_file, list_rois, summary_frame):
    """
    In this problem I will approach the creation of time-traces for Regions Of Interest (ROIs)
    created previously from a summary image, given the list of all frames of the video

    This problem requires for the .tiff file to be passed as numpy (np_file) and a list of binary
    masks of each ROI (list_rois) and and doesn't return anything (None). The summary frame
    (summary_frame) is also given for visualization purposes
    """

    # Part A
    print("PART A")

    # Create intermediate function
    def get_time_trace(np_file, roi):
        """
        This function will output a 1D array of the time-trace over the region of the ROI binary mask,
        being each time point a frame of the video. The method used to characterize one frame is to take the
        mean of the values of all the positive pixels provided the roi mask
        
        This function WON'T check for the validity of the input parameters

        The function requires for the .tiff file to be passed as numpy (np_file) and a binary mask of the ROI
        (roi), which is the same shape as each frame frame of the video, and returns the time-trace array
        (time_trace)
        """

        # Initialize output
        time_trace = []

        # Iterate each frame, and in each frame get mean intensity of ROI
        for frame in np_file:
            # Get frame with ROI imposed
            this_frame_with_roi = frame * roi

            # Get mean intensity of frame over the ROI
            # (mean as the sum of all intensities divided by the number of pixels in ROI)
            sum_intensity = np.sum(this_frame_with_roi)
            n_pixels = np.sum(roi)
            mean_intensity_frame_with_roi = sum_intensity / n_pixels

            # Save value
            time_trace.append(mean_intensity_frame_with_roi)
        
        return time_trace

    # Get time-trace for each roi, and plot it
    for i, roi in enumerate(list_rois):
        # Get time-trace
        time_trace = get_time_trace(np_file, roi)

        print(f"Doing plots of ROI #{i + 1}")

        # Plot roi mask
        plt.figure()
        plt.imshow(summary_frame * roi, cmap="gray")
        plt.axis("off")
        plt.title(f"Summary frame\nwith ROI mask")
        plt.show()

        # Plot time-trace
        plt.figure()
        plt.plot(time_trace)
        plt.title(f"Time-trace of mean intensity of ROI")
        plt.xlabel("Time (frames)")
        plt.ylabel("Mean intensity of ROI (AU)")
        plt.show()

    return None
