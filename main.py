import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

# Define paths and constants
prototxt_path = "models/MobileNetSSD_deploy.prototxt.txt"  # Path to the prototxt file
model_path = "models/MobileNetSSD_deploy.caffemodel"  # Path to the model file
min_confidence = 0.2  # Minimum confidence threshold for object detection
CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
# List of classes that the model can detect
np.random.seed(54321)  # Setting the seed for numpy's random number generator
# Generating random colors for each class
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# Reading the model from Caffe
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Function to perform object detection on the selected image


def detect_objects(image_path):
    image = cv2.imread(image_path)

    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0][0][i][3] * width)
            upper_left_y = int(detected_objects[0][0][i][4] * height)
            lower_right_x = int(detected_objects[0][0][i][5] * width)
            lower_right_y = int(detected_objects[0][0][i][6] * height)

            prediction_text = f"{CLASSES[class_index]}: {confidence:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y),
                          (lower_right_x, lower_right_y), colors[class_index], 2)
            cv2.putText(image, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y >
                                                 30 else upper_left_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[class_index], 2)

    return image

# Function to handle button click event and display detected objects


def process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        output_image = detect_objects(file_path)

        cv2.imshow("Detected Objects", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Create Tkinter application window
root = tk.Tk()  # Creating an instance of the Tk class, representing the main application window

# Overriding the title method inherited from the Tk class to set the title of the window
root.title("Image detection using preexisting model")
# Overriding the geometry method inherited from the Tk class to set the initial size of the window
root.geometry("400x150")

# Create a button to trigger object detection
browse_button = tk.Button(root, text="Select Image", command=process_image)
# Creating an instance of the Button class, which is composed within the root window.
# We're passing root as the parent (container) of the button.
# We're specifying the text displayed on the button as "Select Image".
# We're also specifying the command to be executed when the button is clicked, which is process_image function.

# Calling the pack method to organize the button within the root window.
browse_button.pack(pady=20)
# pady=20 adds vertical padding (space) around the button.

# Running the main event loop of the application to display the window and handle user interactions.
root.mainloop()
