import os
import shutil

from matplotlib import pyplot as plt
import numpy as np
import librosa
import cv2

import subprocess as sp
from threading import Timer

from torchvision.utils import make_grid

import torch
import glob

def convert_to_db(mag, eps=1e-10):
    return 20 * np.log10(np.maximum(mag, eps))

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()


def recover_rgb(img):
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img


def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    # mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    # mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_MAGMA)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


class VideoWriter:
    """ Combine numpy frames into video using ffmpeg

    Arguments:
        filename: name of the output video
        fps: frame per second
        shape: shape of video frame

    Properties:
        add_frame(frame):
            add a frame to the video
        add_frames(frames):
            add multiple frames to the video
        release():
            release writing pipe

    """

    def __init__(self, filename, fps, shape):
        self.file = filename
        self.fps = fps
        self.shape = shape

        # video codec
        ext = filename.split('.')[-1]
        if ext == "mp4":
            self.vcodec = "h264"
        else:
            raise RuntimeError("Video codec not supoorted.")

        # video writing pipe
        cmd = [
            "ffmpeg",
            "-y",                                     # overwrite existing file
            "-f", "rawvideo",                         # file format
            "-s", "{}x{}".format(shape[1], shape[0]), # size of one frame
            "-pix_fmt", "rgb24",                      # 3 channels
            "-r", str(self.fps),                      # frames per second
            "-i", "-",                                # input comes from a pipe
            "-an",                                    # not to expect any audio
            "-vcodec", self.vcodec,                   # video codec
            "-pix_fmt", "yuv420p",                  # output video in yuv420p
            self.file]

        self.pipe = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=10**9)

    def release(self):
        self.pipe.stdin.close()

    def add_frame(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0]
        assert frame.shape[1] == self.shape[1]
        try:
            self.pipe.stdin.write(frame.tostring())
        except:
            _, ffmpeg_error = self.pipe.communicate()
            print(ffmpeg_error)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)


def kill_proc(proc):
    proc.kill()
    print('Process running overtime! Killed.')


def run_proc_timeout(proc, timeout_sec):
    # kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    try:
        cmd = ["ffmpeg", "-y",
               "-loglevel", "quiet",
               "-i", src_video,
               "-i", src_audio,
               "-c:v", "copy",
               "-c:a", "aac",
               "-strict", "experimental",
               dst_video]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, 10.)

        if verbose:
            print('Processed:{}'.format(dst_video))
    except Exception as e:
        print('Error:[{}] {}'.format(dst_video, e))


# save video to the disk using ffmpeg
def save_video(path, tensor, fps=25):
    assert tensor.ndim == 4, 'video should be in 4D numpy array'
    L, H, W, C = tensor.shape
    writer = VideoWriter(
        path,
        fps=fps,
        shape=[H, W])
    for t in range(L):
        writer.add_frame(tensor[t])
    writer.release()


def save_audio(path, audio_numpy, sr):
    librosa.output.write_wav(path, audio_numpy, sr)

import matplotlib.pyplot as plt
import math

MIN, MAX = -12.0, 2.5

def plot_spectrogram(
    spectrogram: torch.Tensor,
    sr=22050, hop_length=256, n_mels=80,
    fmin=0.0, fmax=8000.0,
    title=None, cmap='magma'
):
    """
    spectrogram: [H, W] or [1,H,W] の lnメル（固定レンジで色合わせ）
    """
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.squeeze(0)
    H, W = spectrogram.shape

    # extent: x軸(秒), y軸(Hz) を表示
    time_max = (W - 1) * hop_length / sr
    extent = [0, time_max, fmin, fmax]

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(
        spectrogram.cpu().numpy(),
        aspect="auto", origin="lower", interpolation="nearest",
        vmin=MIN, vmax=MAX, cmap=cmap, extent=extent
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Freq [Hz]")
    if title is not None:
        ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log-mel (ln)")

    plt.tight_layout()
    # TensorBoardに渡した後にメモリ節約
    fig.canvas.draw()
    plt.close(fig)
    return fig

def save_mel_to_tensorboard(batch_data, outputs, writer, epoch, sr=22050, hop=256, fmin=0.0, fmax=11025.0):
    pred = outputs['pred_mag'][:4]  # 例：上位4だけ
    gt   = outputs['gt_mag'][:4]

    for i in range(pred.size(0)):
        pl = plot_spectrogram(pred[i,0], sr, hop, pred.size(-2), fmin, fmax, title=f'pred L #{i}')
        pr = plot_spectrogram(pred[i,1], sr, hop, pred.size(-2), fmin, fmax, title=f'pred R #{i}')
        gl = plot_spectrogram(gt[i,0],   sr, hop, gt.size(-2),   fmin, fmax, title=f'GT L #{i}')
        gr = plot_spectrogram(gt[i,1],   sr, hop, gt.size(-2),   fmin, fmax, title=f'GT R #{i}')

        writer.add_figure(f'pred/left_{i}',  pl, epoch)
        writer.add_figure(f'pred/right_{i}', pr, epoch)
        writer.add_figure(f'gt/left_{i}',    gl, epoch)
        writer.add_figure(f'gt/right_{i}',   gr, epoch)

def save_mel_to_tensorboard2(batch_data, output, writer, epoch):
    pred_mag_imgs = output['pred_mag'][:8]
    gt_mag_imgs = batch_data['binaural_mel'][:8]
    
    
    img_grid = make_grid(pred_mag_imgs, nrow=8)
    writer.add_image('evalimages',img_grid,epoch)
    
    gt_grid = make_grid(gt_mag_imgs, nrow=8)
    writer.add_image('gtimages', gt_grid, epoch)
    
def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '??????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def min_max_normalize(x, x_min=None, x_max=None):
    """
    入力配列 x を min-max 正規化します。
    x_min と x_max が指定されていない場合は、x の最小値と最大値を使用します。
    
    Args:
        x (np.ndarray): 正規化するデータ
        x_min (float, optional): 使用する最小値。指定がなければ x の最小値を使用
        x_max (float, optional): 使用する最大値。指定がなければ x の最大値を使用
    
    Returns:
        x_norm (np.ndarray): 0〜1に正規化されたデータ
        used_min (float): 正規化に実際に使用した最小値
        used_max (float): 正規化に実際に使用した最大値
    """
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def invert_min_max_normalize(x_norm, x_min, x_max):
    """
    min-max 正規化を元のスケールに戻します。
    
    Args:
        x_norm (np.ndarray): 正規化されたデータ（0〜1の範囲）
        x_min (float): 正規化に使用した最小値
        x_max (float): 正規化に使用した最大値
        
    Returns:
        x_original (np.ndarray): 元のスケールに戻したデータ
    """
    x_original = x_norm * (x_max - x_min) + x_min
    return x_original

# 正規化関数（提供された関数そのまま）
def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples