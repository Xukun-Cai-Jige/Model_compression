import av

import numpy as np

from model.utils import read_video_pyav

container = av.open("demo_video/sample_demo_1.mp4")

total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
print(indices)

clip = read_video_pyav(container, indices, "phi3_frames", save=True)