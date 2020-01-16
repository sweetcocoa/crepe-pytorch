import torch
import torch.nn as nn
import torchaudio
import os, sys
from utils import *
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, f, w, s, in_channels):
        super().__init__()
        p1 = (w - 1) // 2
        p2 = (w - 1) - p1
        self.pad= nn.ZeroPad2d((0, 0, p1, p2))
        
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=s)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class CREPE(nn.Module):
    def __init__(self, model_capacity="full"):
        super().__init__()

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        self.layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        for i in range(len(self.layers)):
            f, w, s, in_channel = filters[i+1], widths[i], strides[i], filters[i]
            self.add_module("conv%d" % i, ConvBlock(f, w, s, in_channel))

        self.linear = nn.Linear(64*capacity_multiplier, 360)

        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "crepe-{}.pth".format(model_capacity)
        self.load_state_dict(torch.load(os.path.join(package_dir, filename)))
        self.eval()

    def forward(self, x):
        # x : shape (batch, sample)
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)

        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def get_activation(self, audio, sr, center=True, step_size=10):
        """     
        audio : (N,) or (C, N)
        """

        if sr != 16000:
            rs = torchaudio.transforms.Resample(sr, 16000)
            audio = rs(audio)
        
        if len(audio.shape) == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            else:
                audio = audio.mean(dim=0) # make mono
        
        def get_frame(audio, step_size=10, center=True):
            if center:
                audio = nn.functional.pad(audio, pad=(512, 512))

            # make 1024-sample frames of the audio with hop length of 10 milliseconds
            hop_length = int(16000 * step_size / 1000)
            n_frames = 1 + int((len(audio) - 1024) / hop_length)

            assert audio.dtype == torch.float32
            itemsize = 1 # float32 byte size
            frames = torch.as_strided(audio, size=(1024, n_frames), stride=(itemsize, hop_length * itemsize))
            frames = frames.transpose(0, 1).clone()

            frames = (frames - torch.mean(frames, axis=1).unsqueeze(-1))
            frames /= (torch.std(frames, axis=1).unsqueeze(-1))

            return frames    
        
        
        frames = get_frame(audio, step_size, center)
        
        return self.forward(frames)

    def predict(self, audio, sr, viterbi=False, center=True, step_size=10):
        activation = self.get_activation(audio, sr)
        frequency = to_freq(activation)
        confidence = activation.max(dim=1)[0]
        time = torch.arange(confidence.shape[0]) * step_size / 1000.0
        return time, frequency, confidence, activation

    def process_file(self, file, output=None, viterbi=False, 
                     center=True, step_size=10, save_plot=False):
        try:
            audio, sr = torchaudio.load(file)
        except ValueError:
            print("CREPE-pytorch : Could not read", file, file=sys.stderr)

        with torch.no_grad():
            time, frequency, confidence, activation = self.predict(
                audio, sr, 
                viterbi=viterbi, 
                center=center,
                step_size=step_size
                )

        time, frequency, confidence, activation = time.numpy(), frequency.numpy(), confidence.numpy(), activation.numpy()

        f0_file = os.path.join(output, os.path.basename(os.path.splitext(file)[0])) + ".f0.csv"
        f0_data = np.vstack([time, frequency, confidence]).transpose()
        np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f'], delimiter=',',
                header='time,frequency,confidence', comments='')

        # save the salience visualization in a PNG file
        if save_plot:
            import matplotlib.cm
            from imageio import imwrite

            plot_file = os.path.join(output, os.path.basename(os.path.splitext(file)[0])) + ".activation.png"
            # to draw the low pitches in the bottom
            salience = np.flip(activation, axis=1)
            inferno = matplotlib.cm.get_cmap('inferno')
            image = inferno(salience.transpose())

            imwrite(plot_file, (255 * image).astype(np.uint8))
