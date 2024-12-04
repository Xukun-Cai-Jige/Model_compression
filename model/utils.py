import os
import cv2

import numpy as np


def setup_cache_dir():
    """Setup local cache directories for model and data"""
    base_dir = os.environ.get('BDA_HOME', os.getcwd())
    cache_dir = os.path.join(base_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set HuggingFace cache directory
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
    
    return cache_dir


def extract_frames_video(video_path, output_dir, req_fps=1):
    """
    Extract frames from a video and save them as individual images.
    """
    # Create the output directory if it doesn't exist
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the number of frames to skip based on the desired frame rate
    skip_frames = int(fps / req_fps)

    # Initialize a counter for the number of frames processed
    frames_processed = 0
    # Loop through the video frames
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()
        # If the frame was successfully read
        if ret:
            # If the frame number is divisible by the skip_frames value
            if frames_processed % skip_frames == 0:
                # Save the frame as an image
                frame_path = os.path.join(output_dir, f"frame_{frames_processed}.jpg")
                cv2.imwrite(frame_path, frame)
                # increment the frame counter
                frames_processed += 1
    # end of the video is reached
    cap.release()
    return frames_processed


def read_video_pyav(container, indices, save_path = None, save=False):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    saved_frame_no = 0
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
            if save and save_path is not None:
                frame.to_image().save(f"{save_path}/frame_{saved_frame_no}.jpg")
                saved_frame_no += 1
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])