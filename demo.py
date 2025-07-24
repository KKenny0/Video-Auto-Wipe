# -*- coding: utf-8 -*-
'''
Copyright: Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the model.
Thanks to STTN provider: https://github.com/researchmm/STTN
Author: BUPT_GWY
Contact: a312863063@126.com

@Changed by: KennyWu
@Date: 2025-07-24
@Contact: jdlow@live.cn
'''
import cv2
import numpy as np
import importlib
import argparse
import sys
import torch
import os
import time
from torchvision import transforms
from numba import njit, prange
import concurrent.futures

# My libs
from core.utils import Stack, ToTorchFormatTensor

parser = argparse.ArgumentParser(description="STTN")

parser.add_argument("-t", "--task", type=str, help='CHOOSE THE TASK：delogo or detext', default='detext')
parser.add_argument("-v", "--video", type=str, default='input/detext_examples/chinese1.mp4')
parser.add_argument("-m", "--mask",  type=str, default='input/detext_examples/mask/chinese1_mask.png')
parser.add_argument("-r", "--result",  type=str, default='result/')
parser.add_argument("-d", "--dual",  type=bool, default=False, help='Whether to display the original video in the final video')
parser.add_argument("-w", "--weight",   type=str, default='pretrained_weight/detext_trial.pth')

parser.add_argument("--model", type=str, default='auto-sttn')
parser.add_argument("-g", "--gap",   type=int, default=200, help='set it higher and get result better')
parser.add_argument("-l", "--ref_length",   type=int, default=5)
parser.add_argument("-n", "--neighbor_stride",   type=int, default=5)

args = parser.parse_args()

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def read_frame_info_from_video(vname):
    """
    Read video file and extract frame information.

    Args:
        vname (str): Path to the input video file.

    Returns:
        tuple: A tuple containing:
            - reader: OpenCV VideoCapture object
            - frame_info (dict): Dictionary containing video metadata:
                - W_ori (int): Original width of the video frames
                - H_ori (int): Original height of the video frames
                - fps (float): Frames per second of the video
                - len (int): Total number of frames in the video

    Raises:
        SystemExit: If the video file cannot be opened.
    """
    reader = cv2.VideoCapture(vname)
    if not reader.isOpened():
        print("fail to open video in {}".format(args.input))
        sys.exit(1)
    frame_info = {}
    frame_info['W_ori'] = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    frame_info['H_ori'] = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    frame_info['fps'] = reader.get(cv2.CAP_PROP_FPS)
    frame_info['len'] = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
    return reader, frame_info

def read_mask(path):
    """
    Read and binarize a mask image.

    Args:
        path (str): Path to the mask image file.

    Returns:
        numpy.ndarray: A binary mask with values in {0, 1} and shape (H, W, 1).
    """
    img = cv2.imread(path, 0)
    ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    img = img[:, :, None]
    return img

# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length, args):
    """
    Get reference frame indices for video inpainting.

    This function selects reference frames at regular intervals, excluding
    frames that are in the neighborhood of the current frame.

    Args:
        neighbor_ids (list): List of frame indices to exclude from reference frames.
        length (int): Total number of frames in the video.

    Returns:
        list: List of reference frame indices.
    """
    ref_index = []
    for i in range(0, length, args.ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


@njit(parallel=True)
def blend_frames(comp_frames, pred_img, neighbor_ids, mask):
    for i in prange(len(neighbor_ids)):
        idx = neighbor_ids[i]
        img = pred_img[i].astype(np.float32)
        if not mask[idx]:
            comp_frames[idx] = img
            mask[idx] = True
        else:
            comp_frames[idx] = 0.5 * comp_frames[idx] + 0.5 * img


def process(frames, model, device, w, h, args):
    """
    Process a batch of frames through the inpainting model.

    Args:
        frames (list): List of input frames as numpy arrays.
        model: The inpainting model.
        device: The computation device (CPU/GPU).
        w (int): Width of the output frames.
        h (int): Height of the output frames.

    Returns:
        numpy.ndarray: The processed frames as a numpy array.
    """
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0) * 2 - 1

    feats = feats.to(device)
    comp_frames = [None] * video_length

    with torch.no_grad():
        # accelerate CNN decoding - channels_last
        feats = model.encoder(
            feats.view(video_length, 3, h, w)
            .contiguous()
            .to(memory_format=torch.channels_last)
        )
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)

    # prepare buffer for fused output
    comp_frames_np = np.zeros((video_length, h, w, 3), dtype=np.float32)
    mask = np.zeros((video_length,), dtype=np.bool_)

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, args.neighbor_stride):
        neighbor_ids = [
            i
            for i in range(
                max(0, f - args.neighbor_stride),
                min(video_length, f + args.neighbor_stride + 1),
            )
        ]
        ref_ids = get_ref_index(neighbor_ids, video_length, args)

        # Inference and decode
        with torch.no_grad():
            # amp acceleration
            with torch.cuda.amp.autocast():
                ids = neighbor_ids + ref_ids
                input_feats = (
                    feats[0, ids, :, :, :]
                    .contiguous()
                    .to(memory_format=torch.channels_last)
                )
                pred_feat = model.infer(input_feats)
                decoded = model.decoder(
                    pred_feat[: len(neighbor_ids), :, :, :]
                ).detach()
                pred_img = torch.tanh(decoded)

            # post-process: scale to [0, 255]
            pred_img = ((pred_img + 1.0) * 127.5).clamp(0, 255)
            # pre type conversion
            pred_img = (
                pred_img.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
            )

        # numba acceleration
        blend_frames(comp_frames_np, pred_img, np.array(neighbor_ids), mask)
        comp_frames = []
        for idx in range(video_length):
            if mask[idx]:
                comp_frames.append(
                    np.clip(comp_frames_np[idx], 0, 255).astype(np.uint8)
                )
            else:
                comp_frames.append(None)
    return comp_frames


def get_inpaint_mode_for_detext(H, h, mask):
    """
    Get inpainting mode for text removal task.

    This function determines the inpainting mode based on the mask position.

    Args:
        H (int): Height of the frame.
        h (int): Height of the mask.
        mask: The binary mask.

    Returns:
        str: The inpainting mode ('top', 'bottom', or 'center').
    """  # get inpaint segment
    mode = []
    to_H = from_H = H  # the subtitles are usually underneath
    while from_H != 0:
        if to_H - h < 0:
            from_H = 0
            to_H = h
        else:
            from_H = to_H - h
        if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
            if to_H != H:
                move = 0
                while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                    move += 1
                if to_H + move < H and move < h:
                    to_H += move
                    from_H += move
            mode.append((from_H, to_H))
        to_H -= h
    return mode


class InpaintingTask:
    """
    Abstract base class for an inpainting task.

    This class defines the interface for different inpainting tasks, such as
    text removal or logo removal. Subclasses should implement the specific
    logic for each step of the inpainting process.
    """

    def __init__(self, args):
        self.args = args
        self.model = None
        self.device = None
        self.reader = None
        self.frame_info = None
        self.mask = None
        self.video_length = 0
        self.h = 0
        self.w = 0

    def pre_process(self):
        """Prepare the model and data for video inpainting."""
        print(f"Task: {self.args.task}")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = importlib.import_module("model." + self.args.model)
        self.model = net.InpaintGenerator().to(self.device)
        data = torch.load(self.args.weight, map_location=self.device)
        self.model.load_state_dict(data["netG"])
        self.model.eval()
        print(f"Loading weight from: {self.args.weight}")

        self.reader, self.frame_info = read_frame_info_from_video(self.args.video)
        print(f"Loading video from: {self.args.video}")

        if not os.path.exists(self.args.result):
            os.makedirs(self.args.result)

        self.mask = read_mask(self.args.mask)
        print(f"Loading mask from: {self.args.mask}")

        self.video_length = self.frame_info["len"]
        self.h, self.w = self.frame_info["H_ori"], self.frame_info["W_ori"]

    def process_video(self):
        """Process the entire video, frame by frame."""
        raise NotImplementedError("Subclasses must implement process_video")

    def run(self):
        """Execute the inpainting task."""
        start_time = time.time()
        self.pre_process()
        self.process_video()
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f}s")


class DetextTask(InpaintingTask):
    """Inpainting task for removing text from videos."""

    def process_video(self):
        w, h = 640, 120
        video_name = os.path.basename(self.args.video).split(".")[0]
        video_out_path = os.path.join(
            self.args.result, f"{video_name}_{self.args.task}.mp4"
        )
        writer = cv2.VideoWriter(
            video_out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.frame_info["fps"],
            (self.w, self.h) if not self.args.dual else (self.w, self.h * 2),
        )

        split_h = int(self.w * 3 / 16)
        mode = get_inpaint_mode_for_detext(self.h, split_h, self.mask)

        rec_time = (
            self.video_length // self.args.gap
            if self.video_length % self.args.gap == 0
            else self.video_length // self.args.gap + 1
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(mode)) as executor:
            for i in range(rec_time):
                start_f = i * self.args.gap
                end_f = min((i + 1) * self.args.gap, self.video_length)
                print(f"Processing frames {start_f + 1}-{end_f}/{self.video_length}")

                frames_hr = []
                frames = {k: [] for k in range(len(mode))}
                comps = {}

                # read frames
                read_start_time = time.time()
                for j in range(start_f, end_f):
                    success, image = self.reader.read()
                    if not success:
                        break
                    frames_hr.append(image)
                    for k in range(len(mode)):
                        image_crop = image[mode[k][0] : mode[k][1], :, :]
                        image_resize = cv2.resize(image_crop, (w, h))
                        frames[k].append(image_resize)
                read_end_time = time.time()
                print(f"Read frames time: {read_end_time - read_start_time:.2f}s")

                if not frames_hr:
                    break

                # process frames
                process_start_time = time.time()
                futures = {
                    k: executor.submit(
                        process, frames[k], self.model, self.device, w, h, self.args
                    )
                    for k in range(len(mode))
                    if frames[k]
                }
                comps = {k: futures[k].result() for k in futures}
                process_end_time = time.time()
                print(
                    f"Process frames time: {process_end_time - process_start_time:.2f}s"
                )

                # write frames
                write_start_time = time.time()
                for j in range(len(frames_hr)):
                    frame_ori = frames_hr[j].copy()
                    frame = frames_hr[j]
                    for k in range(len(mode)):
                        if comps.get(k) and j < len(comps[k]):
                            comp = cv2.resize(comps[k][j], (self.w, split_h))
                            comp = cv2.cvtColor(
                                np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB
                            )
                            mask_area = self.mask[mode[k][0] : mode[k][1], :]
                            frame[mode[k][0] : mode[k][1], :, :] = (
                                mask_area * comp
                                + (1 - mask_area) * frame[mode[k][0] : mode[k][1], :, :]
                            )
                    if self.args.dual:
                        frame = np.vstack([frame_ori, frame])
                    writer.write(frame)
                write_end_time = time.time()
                print(f"Write frames time: {write_end_time - write_start_time:.2f}s")

        writer.release()
        self.reader.release()
        print("--------------------------------------")
        print(f"Finished processing. Saved to {video_out_path}")


class DelogoTask(InpaintingTask):
    """Inpainting task for removing logos from videos."""

    def process_video(self):
        # Implementation for logo removal would go here.
        # This is just a placeholder.
        print("Processing video for logo removal...")
        # Similar video processing loop as in DetextTask but with logo-specific logic.
        pass


def task_factory(args):
    """Factory function to create an inpainting task based on arguments."""
    if args.task == "detext":
        return DetextTask(args)
    elif args.task == "delogo":
        return DelogoTask(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


def main(args):
    """
    Main function for the video inpainting demo.

    This function handles the command-line interface, processes the input video
    using the specified model, and saves the output video with inpainted regions.

    The script supports both logo removal and text removal tasks, with options
    to control the inpainting process and output format.
    """
    task = task_factory(args)
    task.run()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
