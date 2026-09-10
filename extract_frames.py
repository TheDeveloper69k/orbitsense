import cv2
import os
import sys


def extract_frames(video_path, output_folder, every_n_frames=10):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_count % every_n_frames == 0:
            filename = os.path.join(
                output_folder,
                f"frame_{saved_count:04d}.jpg"
            )

            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Done! Saved {saved_count} images to:")
    print(output_folder)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage:")
        print("python extract_frames.py video.mp4 output_folder")
        sys.exit()

    video_path = sys.argv[1]
    output_folder = sys.argv[2]

    extract_frames(
        video_path,
        output_folder,
        every_n_frames=10
    )