import os
import cv2
import av
import threading
import subprocess
from queue import Queue

"""
Given individual video files (mp4, webm) on disk, creates a folder for
every video file and saves the video's RGB frames as jpeg files in that
folder.

It can be used to turn clips/, which comes as
many ".mp4" files, into an RGB folder for each ".mp4" file.
Uses multithreading to extract frames faster.

Modify the two filepaths at the bottom and then run this script.
"""


FILE_TEMPLATE = 'frame_{:012d}.jpg'
OUT_HEIGHT_WIDTH = (398, 224) # approximately 16:9 aspect ratio
VIDEO_PATH = '/home/rocus/Documents/john/grepl/clips/'  # Path to the folder containing video files
RGB_OUT_PATH = '/home/rocus/Documents/john/grepl/clip_frames' # the root output path where RGB frame folders should be created
FILE_EXTENSION = '.mp4'  # File extension of the videos to process

FPS_OUT = 10


def video_to_rgb(video_filename: str, out_dir: str, resize_shape: tuple, fps_out: int = FPS_OUT):
    """Convert a video file to a series of RGB frames.
    Extracts frames from the video and saves them as JPEG images in the specified output directory.
    Starts with OpenCV for efficiency, and falls back to PyAV if OpenCV fails.
    
    TODO specify frame rate?

    Args:
        video_filename (str): The path to the video file.
        out_dir (str): The output directory where frames will be saved.
        resize_shape (tuple): The target size for the frames (width, height).

    Returns:
        None: The function saves frames to the specified directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- try OpenCV first ----------------------------------------------------
    cap = cv2.VideoCapture(video_filename)
    ok, frame = cap.read()
    if ok:
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the frame interval more precisely
        frame_interval = fps_in / fps_out
        next_frame_to_save = 0
        idx, frames_captured = 0, 0
        while ok:
            if idx >= next_frame_to_save:
                cv2.imwrite(os.path.join(out_dir, FILE_TEMPLATE.format(frames_captured)),
                            cv2.resize(frame, resize_shape))
                frames_captured += 1
                next_frame_to_save += frame_interval
            ok, frame = cap.read()
            idx += 1
        cap.release()
        print(f"Extracted {frames_captured} frames at {fps_out} fps from {video_filename} to {out_dir}")
        return                      # success, we're done

    print("OpenCV failed, falling back to PyAV...")
    # --- fallback: PyAV (uses system FFmpeg → AV1 works) ---------------------
    with av.open(video_filename) as container:
        # Get input video stream information
        if len(container.streams.video) > 0 and (fps_in := container.streams.video[0].average_rate):
            # Same frame interval calculation as above
            frame_interval = fps_in / fps_out
            next_frame_to_save = 0
        else:
            # Fallback to saving every frame if we can't determine input fps
            frame_interval = 1
            next_frame_to_save = 0
        
        frames_captured = 0
        for idx, frame in enumerate(container.decode(video=0)):
            if idx >= next_frame_to_save:
                img = cv2.resize(frame.to_ndarray(format='bgr24'), resize_shape)
                cv2.imwrite(os.path.join(out_dir, FILE_TEMPLATE.format(frames_captured)), img)
                frames_captured += 1
                next_frame_to_save += frame_interval
    return

def process_videofile(video_filename, video_path, rgb_out_path, file_extension: str ='.mp4'):
    filepath = os.path.join(video_path, video_filename)
    video_filename = video_filename.replace(file_extension, '')

    out_dir = os.path.join(rgb_out_path, video_filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    video_to_rgb(filepath, out_dir, resize_shape=OUT_HEIGHT_WIDTH)

def thread_job(queue, video_path, rgb_out_path, file_extension='.webm'):
    while not queue.empty():
        video_filename = queue.get()
        process_videofile(video_filename, video_path, rgb_out_path, file_extension=file_extension)
        queue.task_done()


if __name__ == '__main__':

    video_filenames = os.listdir(VIDEO_PATH)
    queue = Queue()
    [queue.put(video_filename) for video_filename in video_filenames if video_filename.endswith(FILE_EXTENSION)]

    NUM_THREADS = 15
    for i in range(NUM_THREADS):
        worker = threading.Thread(target=thread_job, args=(queue, VIDEO_PATH, RGB_OUT_PATH, FILE_EXTENSION))
        worker.start()

    print('waiting for all videos to be completed.', queue.qsize(), 'videos')
    print('This can take an hour or two depending on dataset size')
    queue.join()
    print('all done')