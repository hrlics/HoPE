import base64
from io import BytesIO
from random import sample
from typing import Optional, Tuple, Union

import av
import numpy as np
from av.codec.context import CodecContext
from decord import VideoReader, cpu
from PIL import Image

import math
from PIL.Image import Image as ImageObject

# MODIFIED
# in order to support fps
def load_video_decord(video_path, max_frames_num, fps):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    
    real_fps = vr.get_avg_fps()
    sample_frames = float(total_frame_num / real_fps) * fps
    sample_frames = min(total_frame_num, max_frames_num, sample_frames)
    sample_frames = math.floor(sample_frames)
    sample_indices = np.linspace(0, total_frame_num - 1, sample_frames).astype(np.int32)

    batch_frames = vr.get_batch(sample_indices).asnumpy()
    frames = [Image.fromarray(frame) for frame in batch_frames]
    return frames  # (frames, height, width, channels)


class Qwen2vlPlugin():
    def __init__(self, image_token: Optional[str], video_token: Optional[str], **kwargs):
        # super().__init__(image_token=image_token, video_token=video_token, **kwargs)
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200

        self.VIDEO_MIN_PIXELS = 128 * 28 * 28
        self.VIDEO_MAX_PIXELS = 768 * 28 * 28
        self.VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
        self.FRAME_FACTOR = 2
        self.FPS = 2.0
        self.FPS_MIN_FRAMES = 4
        self.FPS_MAX_FRAMES = 768


    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor


    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor


    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(
        self, height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if max(height, width) / min(height, width) > self.MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        if image.mode != "RGB":
            image = image.convert("RGB")
        min_pixels = kwargs.get("min_pixels", self.VIDEO_MIN_PIXELS)
        total_pixels = kwargs.get("total_pixels", self.VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(self.VIDEO_MAX_PIXELS, int(total_pixels // kwargs.get('images_len', 64)) * self.FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = kwargs.get("max_pixels", max_pixels)
        resized_height, resized_width = self.smart_resize(
            image.height,
            image.width,
            factor=self.IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        image = image.resize((resized_width, resized_height), resample=Image.NEAREST)

        return image

def read_video_decord(video_path, max_frames_num, fps, total_pixels):
    results = []
    images = load_video_decord(video_path, max_frames_num, fps)
    helper = Qwen2vlPlugin(image_token=None, video_token=None)

    images_len = len(images)

    for image in images:
        if isinstance(image, str):
                image = Image.open(image)
        elif isinstance(image, dict):
            if image["bytes"] is not None:
                image = Image.open(BytesIO(image["bytes"]))
            else:
                image = Image.open(image["path"])
        
        if not isinstance(image, ImageObject):
                raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))
        
        results.append(helper.preprocess_image(image, total_pixels=total_pixels, images_len=images_len))

    return results


# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def load_video_stream(container, num_frm: int = 8, fps: float = None, force_include_last_frame=False):
    # container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    frame_rate = container.streams.video[0].average_rate
    if fps is not None:
        video_length = total_frames / frame_rate
        num_frm = min(num_frm, int(video_length * fps))
    sampled_frm = min(total_frames, num_frm)
    indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
    if force_include_last_frame:
        last_frame = total_frames - 1
        if last_frame not in indices:
            indices = np.linspace(0, total_frames - 2, sampled_frm - 1, dtype=int)
            indices = np.append(indices, last_frame)

    return record_video_length_stream(container, indices)


def load_video_packet(container, num_frm: int = 8, fps: float = None):
    frames = record_video_length_packet(container)
    total_frames = len(frames)
    frame_rate = container.streams.video[0].average_rate
    if fps is not None:
        video_length = total_frames / frame_rate
        num_frm = min(num_frm, int(video_length * fps))
    sampled_frm = min(total_frames, num_frm)
    indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)

    # Append the last frame index if not already included
    if total_frames - 1 not in indices:
        indices = np.append(indices, total_frames - 1)

    return [frames[i] for i in indices]


def read_video_pyav(video_path: str, *, num_frm: int = 8, fps: float = None, format="rgb24", force_include_last_frame=False) -> np.ndarray:
    """
    Read video using the PyAV library.

    Args:
        video_path (str): The path to the video file.
        num_frm (int, optional): The maximum number of frames to extract. Defaults to 8.
        fps (float, optional): The frames per second for extraction. If `None`, the maximum number of frames will be extracted. Defaults to None.
        format (str, optional): The format of the extracted frames. Defaults to "rgb24".

    Returns:
        np.ndarray: A numpy array containing the extracted frames in RGB format.
    """

    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        # For mp4, we try loading with stream first
        try:
            frames = load_video_stream(container, num_frm, fps, force_include_last_frame=force_include_last_frame)
        except:
            frames = record_video_length_packet(container)
    else:
        frames = record_video_length_packet(container)

    return np.stack([x.to_ndarray(format=format) for x in frames])


def read_video_pyav_pil(video_path: str, *, num_frm: int = 8, fps: float = None, format="rgb24", max_image_size: Optional[Union[Tuple[int, int], int]] = None, resize_strategy: str = "resize", force_include_last_frame=False):
    frames = read_video_pyav(video_path, num_frm=num_frm, fps=fps, format=format, force_include_last_frame=force_include_last_frame)
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max_image_size:
            if resize_strategy == "resize":
                if isinstance(max_image_size, int):
                    max_image_size = (max_image_size, max_image_size)
                img = img.resize(max_image_size)
            elif resize_strategy == "thumbnail":
                img.thumbnail(max_image_size)
            else:
                raise ValueError(f"Unknown resize strategy: {resize_strategy}")
        pil_frames.append(img)
    return pil_frames
    # return [Image.fromarray(frame) for frame in frames]


def read_video_pyav_base64(video_path: str, *, num_frm: int = 8, fps: Optional[float] = None, format="rgb24", img_format="PNG", max_image_size: Optional[Union[Tuple[int, int], int]] = None, resize_strategy: str = "resize"):
    frames = read_video_pyav(video_path, num_frm=num_frm, fps=fps, format=format)
    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max_image_size:
            if resize_strategy == "resize":
                if isinstance(max_image_size, int):
                    max_image_size = (max_image_size, max_image_size)
                img = img.resize(max_image_size)
            elif resize_strategy == "thumbnail":
                img.thumbnail(max_image_size)
            else:
                raise ValueError(f"Unknown resize strategy: {resize_strategy}")
        output_buffer = BytesIO()
        img.save(output_buffer, format=img_format)
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        base64_frames.append(base64_str)
    return base64_frames