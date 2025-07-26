import subprocess
import io
from PIL import Image, UnidentifiedImageError

def get_frame_from_timestamp(video_path, timestamp) -> Image:
    """
    Get a frame from a video at a given timestamp and return it as a PIL Image.
    """
    # Use ffmpeg pipe to extract a single frame (supports AV1)
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        img = Image.open(io.BytesIO(proc.stdout))
        return img.convert("RGB")
    except subprocess.CalledProcessError:
        raise ValueError(f"Failed to extract frame at {timestamp}s via ffmpeg")
    except UnidentifiedImageError:
        raise ValueError(f"Failed to extract frame at {timestamp}s via ffmpeg (clip too short?)")

# Example usage
if __name__ == "__main__":
    # Path to the video file
    video_path = '../data/raw_film/7181741-gordon-ryan-vs-philip-rowe-wno-the-return-of-gordon-ryan.mp4'
    # Timestamp in seconds
    timestamp = 609  # e.g., 10 seconds

    # Get the frame from the specified timestamp
    img = get_frame_from_timestamp(video_path, timestamp)
    # Display the image
    img.show()