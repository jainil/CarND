import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    xy_window, xy_overlap = np.array(xy_window), np.array(xy_overlap)
    # If x and/or y start/stop positions not defined, set to image size
    x_start = x_start_stop[0] or 0
    x_stop = x_start_stop[1] or img.shape[1]
    y_start = y_start_stop[0] or 0
    y_stop = y_start_stop[1] or img.shape[0]
    # Compute the span of the region to be searched
    span = np.array([x_stop - x_start, y_stop - y_start])
    # Compute the number of pixels per step in x/y
    px_per_step = np.asarray(xy_window * (1 - xy_overlap), dtype=int )
    # Compute the number of windows in x/y
    windows = np.asarray( span/px_per_step - 1 , dtype=int )
    # Initialize a list to append window positions to
    window_list = []
    for x in range(windows[0]):
        for y in range(windows[1]):
            p1 = np.array([x, y]) * px_per_step
            window_list.append((tuple(p1), tuple(p1 + xy_window)))
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    return window_list

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))

window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)