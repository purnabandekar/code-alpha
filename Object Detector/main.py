
import cv2
import argparse
from ultralytics import YOLO

# ----------------------------------
# OBJECT DETECTION AND TRACKING
# ----------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection and Tracking"
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Use 0 for webcam or provide video file path"
    )

    args = parser.parse_args()

    # Load pretrained YOLO model
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    # Select video source
    source = 0 if args.source == "0" else args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Object Detection and Tracking Started!")
    print("Press Q to exit.")

    while True:

        success, frame = cap.read()

        if not success:
            print("Video ended or frame unavailable.")
            break

        # Detect and track objects
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes and tracking IDs
        annotated_frame = result.plot()

        # Count detected objects
        object_count = len(result.boxes)

        # Display object count
        cv2.putText(
            annotated_frame,
            f"Objects Detected: {object_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display tracking information
        if result.boxes is not None:

            for box in result.boxes:

                if box.id is not None:

                    track_id = int(box.id.item())
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())

                    object_name = model.names[class_id]

                    print(
                        f"ID: {track_id} | "
                        f"Object: {object_name} | "
                        f"Confidence: {confidence:.2f}"
                    )

        # Display output
        cv2.imshow(
            "YOLO Object Detection and Tracking",
            annotated_frame
        )

        # Exit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Tracking stopped.")


if __name__ == "__main__":
    main()
