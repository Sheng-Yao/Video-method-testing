import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# # Export YOLO model into NCNN format (Only requires to run once)
# # Load a YOLO model
# model = YOLO("yolo11n.pt")

# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the exported NCNN model
ncnn_model = YOLO("yolo11n_ncnn_model")

# Run inference
while True:
# Capture frame-by-frame
    frame = picam2.capture_array()

    # Run YOLO11 inference on the frame
    results = ncnn_model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()