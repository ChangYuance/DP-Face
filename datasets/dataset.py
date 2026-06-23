import os
import torch
import glob
import numpy as np
import csv
import random
import PIL.Image as Image
import torchvision
from torch.utils import data
import torchaudio
from .video_transform import *
import torchaudio.transforms as T
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PalsyDataset(data.Dataset):
    def __init__(self, args, mode):
        """Dataset for Palsy-330

        Args:
            args
            mode: String("train" or "test")

            num_frames: the number of sampled frames from every video, default: 16
            image_size: crop images to 112*112

        """
        self.args = args
        self.path = self.args.train_dataset if mode == "train" else (self.args.val_dataset if mode == "val" else self.args.test_dataset)
        if self.args.use_test and mode == "test":
            self.path = self.path.replace("test", "val")
        self.num_frames = self.args.num_frames
        self.image_size = self.args.crop_size
        self.mode = mode
        self.transform = self.get_transform()
        self.data = self.get_data()

        pass

    def get_data(self):
        """get data path, label from the csv file

        Returns:
            data_dict:{"path", "label", "num_frames"}
        """
        full_data = []

        npy_path = self.path.replace('csv', 'npy')
        print("loading data")
        # save/load the data to/from npy file
        if os.path.exists(npy_path):
            full_data = np.load(npy_path, allow_pickle=True)
        else:
            with open(self.path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    path = row[0]
                    label = int(row[1]) - 1
                    # modify the path
                    while len(path) < 5:
                        path = "0" + path
                    # combine the path
                    path = os.path.join(
                        self.args.root, "Clip/clip_224x224/", path)
                    full_num_frames = len(os.listdir(path))
                    # get the paths of the frames of a video and sort
                    full_video_frames_paths = glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.png'))
                    full_video_frames_paths.sort()
                    full_data.append({"path": full_video_frames_paths,
                                    "label": label,
                                    "num_frames": full_num_frames})

                np.save(npy_path, full_data)

        print("data loaded")
        return full_data

    def get_transform(self):
        """get trasform accorging to train/test mode and args including: crop, flip, color jitter

        Returns:
            transform
        """
        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                        GroupRandomHorizontalFlip(),
                                                        GroupColorJitter(
                                                            self.args.color_jitter),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
            # transform = torchvision.transforms.Compose([
            #     GroupRandomSizedCrop(self.image_size),
            #     GroupRandomHorizontalFlip(),
            #     Stack(),
            #     ToTorchFormatTensor()
            # ])

        elif self.mode == "test" or self.mode == "val":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        return transform

    def __getitem__(self, index):

        # get the data according to index
        data = self.data[index]
        full_video_frames_paths = data['path']
        video_frames_paths = []
        full_num_frames = len(full_video_frames_paths)
        # 获取文件路径
        from pathlib import Path
        p = Path(data['path'][0])
        last_three = "/".join(p.parts[-4:-1])  # ← 最推荐！简洁、可读、无副作用
        # when getting the frames, randomly choose the neighbour to augment
        for i in range(self.num_frames):
            frame = int(full_num_frames * i / self.num_frames)
            if self.args.random_sample:
                frame += int(random.random() * self.num_frames)
                frame = min(full_num_frames - 1, frame)
            video_frames_paths.append(full_video_frames_paths[frame])

        # get the images and transform
        images = []
        AUs = []
        for video_frames_path in video_frames_paths:
            images.append(Image.open(video_frames_path).convert('RGB'))
            if self.args.AUs:
                npy_path = video_frames_path.replace(".png", ".npy")
                AU = torch.from_numpy(np.load(npy_path)).float()
                AUs.append(AU)
        if self.args.audio:
            audio_path = video_frames_path.replace("landmark_overlay", "audio")
            audio_path = os.path.dirname(os.path.dirname(audio_path))
            audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))]
            waveform, sample_rate = torchaudio.load(audio_files[0])
            transform = T.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,      # 适中
                hop_length=107,   # 4s -> 600 帧
                n_mels=128
            )
            mel_spectrogram = transform(waveform)
            if mel_spectrogram.shape[2] < 600:
            # 填充
                mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, 600 - mel_spectrogram.shape[2]))
            elif mel_spectrogram.shape[2] > 600:
                # 裁剪
                mel_spectrogram = mel_spectrogram[:, :, :600]
            mel_spectrogram = (mel_spectrogram + 4.26) / (4.57 * 2)
            mel_spectrogram = mel_spectrogram.permute(0, 2, 1)
        if self.args.AUs:
            AUs = torch.stack(AUs, dim=0)
        images = self.transform(images)
        images = torch.reshape(
            images, (-1, 3, self.image_size, self.image_size))
        if self.args.audio:
            return images, mel_spectrogram, last_three, data["label"]# torch.Size([16, 3, 224, 224])
        if self.args.AUs:
            return images, AUs, last_three, data["label"]
        else:
            return images, last_three, data["label"]

    def __len__(self):
        return len(self.data)
