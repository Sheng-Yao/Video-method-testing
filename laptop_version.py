# Import opencv2 module
import cv2
# Import YOLO module for object detection
from ultralytics import YOLO

# # Export YOLO model into NCNN format (Only requires to run once)
# # Load a YOLO model
# model = YOLO("yolo11n.pt")

# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Open the default camera
cam = cv2.VideoCapture(0)

ncnn_model = YOLO("yolo11n_ncnn_model")

# Specify the class to predict 
# ncnn_model.predict(classes=0)

while True:
    # Capture the video by frame
    ret, frame = cam.read()

    # Send in the frame to YOLO model for object detection
    results = ncnn_model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()