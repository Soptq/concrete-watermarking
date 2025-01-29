import logging

import configs
import utils
import torch
from torch import nn, Tensor
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat
import torchvision
from torch.nn import functional as thf
import torchvision.transforms as transforms
import bchlib

n_bits = 6
qlinear_args = {
    "weight_bit_width": n_bits,
    "weight_quant": Int8WeightPerTensorFloat,
    "bias": True,
    "bias_quant": None,
    "narrow_range": True
}
logger = logging.getLogger(__name__)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = thf.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class QuantUpsample(nn.Module):
    def __init__(self, scale_factor):
        super(QuantUpsample, self).__init__()
        self.s = scale_factor

    def forward(self, x):
        n, c, h, w = x.shape
        out = x.reshape(-1, c, h, 1, w, 1)
        out = torch.cat([out] * self.s, dim=-3)
        out = torch.cat([out] * self.s, dim=-1)
        out = out.reshape(-1, c, h * self.s, w * self.s)
        return out


class ImageReshapeLayer(nn.Module):
    def __init__(self, hidden_dim=16, channel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel

    def forward(self, x):
        return x.reshape(-1, self.channel, self.hidden_dim, self.hidden_dim)


class QuantImageRepeatLayer2R(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat([x, x], dim=-1)
        x = torch.cat([x, x], dim=-2)
        return x


class Reflection1xPad2d(nn.Module):
    def __init__(self):
        super(Reflection1xPad2d, self).__init__()

    def forward(self, x):
        x_pad_right = x[:, :, :, [-2]]
        x_pad = torch.cat([x, x_pad_right], dim=3)

        x_pad_bottom = x_pad[:, :, [-2], :]
        x_pad = torch.cat([x_pad, x_pad_bottom], dim=2)

        return x_pad


class QuantWatermark2Image(nn.Module):
    def __init__(self, watermark_len, resolution=32, hidden_dim=8, channel=3):
        super().__init__()
        assert resolution % hidden_dim == 0, "Resolution should be divisible by hidden_dim"
        pad_length = resolution // 4
        self.transform = nn.Sequential(
            # [bsz, 32] -> [bsz, 8 * 8 * 3]
            qnn.QuantLinear(watermark_len, hidden_dim * hidden_dim * channel, **qlinear_args),
            # [bsz, 8 * 8 * 3] -> [bsz, 3, 8, 8]
            ImageReshapeLayer(hidden_dim),
            # [bsz, 3, 8, 8] -> [bsz, 3, 16, 16]
            QuantImageRepeatLayer2R(),
            # [bsz, 3, 16, 16] -> [bsz, 3, 32, 32]
            QuantUpsample(scale_factor=2),
            # nn.ZeroPad2d(pad_length),
            # QuantImagePadding8Pixel(),
            qnn.QuantReLU(bit_width=n_bits)
        )

    def forward(self, x):
        return self.transform(x)


class QuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activ='relu'):
        super().__init__()
        self.quant_input = qnn.QuantIdentity(bit_width=n_bits)
        self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    **qlinear_args)
        if activ == 'relu':
            self.activ = qnn.QuantReLU(bit_width=n_bits)
        elif activ == 'tanh':
            self.activ = qnn.QuantTanh(bit_width=n_bits)
        else:
            self.activ = None

    def forward(self, x):
        x = self.quant_input(x)
        x = self.conv(x)
        if self.activ:
            x = self.activ(x)
        return x


class QuantDecBlock(nn.Module):
    def __init__(self, in_channels, skip_channels='default'):
        super().__init__()
        if skip_channels == 'default':
            skip_channels = in_channels // 2
        out_channels = in_channels // 2
        self.up = QuantUpsample(scale_factor=2)
        self.pad = Reflection1xPad2d()
        self.conv1 = QuantConv2d(in_channels, out_channels, 2, 1, 0)
        self.conv2 = QuantConv2d(out_channels + skip_channels, out_channels, 3, 1, 1)
        self.quant_ind = qnn.QuantIdentity(bit_width=n_bits)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.quant_ind(x)
        skip = self.quant_ind(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config

        # input quant
        self.quant_image = qnn.QuantIdentity(bit_width=n_bits)
        self.quant_watermark = qnn.QuantIdentity(bit_width=n_bits)
        self.quant_dec = qnn.QuantIdentity(bit_width=n_bits)

        self.watermark2image = QuantWatermark2Image(config.num_encoded_bits, config.image_shape[0],
                                                    config.watermark_hidden_dim)
        # input_channel: 3 from image + 3 from watermark
        self.pre = QuantConv2d(6, config.num_initial_channels, 3, 1, 1)
        self.enc = nn.ModuleList()
        input_channel = config.num_initial_channels
        for _ in range(config.num_down_levels):
            self.enc.append(QuantConv2d(input_channel, input_channel * 2, 3, 2, 1))
            input_channel *= 2

        self.dec = nn.ModuleList()
        for i in range(config.num_down_levels):
            skip_width = input_channel // 2 if i < config.num_down_levels - 1 else input_channel // 2 + 6
            self.dec.append(QuantDecBlock(input_channel, skip_width))
            input_channel //= 2

        self.post = nn.Sequential(
            QuantConv2d(input_channel, input_channel, 3, 1, 1, activ='none'),
            QuantConv2d(input_channel, input_channel // 2, 1, 1, 0, activ='relu'),
            QuantConv2d(input_channel // 2, 3, 1, 1, 0, activ='tanh')
        )

    def forward(self, image: torch.Tensor, watermark=None):
        if watermark is None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device=image.device)

        # quant input
        watermark = self.quant_watermark(watermark)
        watermark = self.watermark2image(watermark)
        image = self.quant_image(image)
        watermark = self.quant_image(watermark)
        inputs = torch.cat([image, watermark], dim=1)

        enc = []
        x = self.pre(inputs)
        for layer in self.enc:
            enc.append(x)
            x = layer(x)

        enc = enc[::-1]
        for i, (layer, skip) in enumerate(zip(self.dec, enc)):
            if i < self.config.num_down_levels - 1:
                x = layer(x, skip)
            else:
                skip = self.quant_dec(skip)
                inputs = self.quant_dec(inputs)
                x = layer(x, torch.cat([skip, inputs], dim=1))
        return self.post(x)


class DisResNet(nn.Module):
    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config
        self.extractor = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features, config.num_classes - 1)
        self.main = nn.Sequential(
            self.extractor,
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor):
        if image.shape[-1] != self.config.image_shape[-1] or image.shape[-2] != self.config.image_shape[-2]:
            logger.debug(f"Image shape should be {self.config.image_shape} but got {image.shape}")
            image = transforms.Resize(self.config.image_shape)(image)
        return self.main(image)


class Extractor(nn.Module):
    def __init__(self, config: configs.ModelConfig):
        super().__init__()
        self.config = config

        self.extractor = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        n_inputs = None
        for name, child in self.extractor.named_children():
            if name == 'classifier':
                for sub_name, sub_child in child.named_children():
                    if sub_name == '2':
                        n_inputs = sub_child.in_features

        self.extractor.classifier = nn.Sequential(
            LayerNorm2d(n_inputs, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(in_features=n_inputs, out_features=config.num_encoded_bits),
        )

        self.main = nn.Sequential(
            self.extractor,
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor):
        if image.shape[-1] != self.config.image_shape[-1] or image.shape[-2] != self.config.image_shape[-2]:
            logger.debug(f"Image shape should be {self.config.image_shape} but got {image.shape}")
            image = transforms.Resize(self.config.image_shape)(image)
        return self.main(image)


class BCHECC:
    def __init__(self, t, m):
        self.t = t  # number of errors to be corrected
        self.m = m  # total of bits n is 2^m
        self.bch = bchlib.BCH(t, m=m)
        self.data_bytes = (self.bch.n + 7) // 8 - self.bch.ecc_bytes

    def batch_encode(self, batch_size):
        secrets = []
        uuid_bytes = utils.uuid_to_bytes(batch_size)
        for input in uuid_bytes:
            ecc = self.bch.encode(input)
            secrets += [torch.Tensor([int(i) for i in ''.join(format(x, '08b') for x in input + ecc)])]
            assert len(secrets[-1]) == 2 ** self.m, f"Encoded secret bits length should be {2 ** self.m}"
        return torch.vstack(secrets).type(torch.float32)

    def batch_decode_ecc(self, secrets: torch.Tensor, threshold: float = 0.5):
        res = []
        for i in range(len(secrets)):
            packet = self._bch_correct(secrets[i], threshold)
            data_bits = [int(k) for k in ''.join(format(x, '08b') for x in packet)]
            res.append(torch.Tensor(data_bits).type(torch.float32))
        return torch.vstack(res)

    def encode_str(self, input: str):
        assert len(input) == self.data_bytes, f"Input str length should be {self.data_bytes}"
        input_bytes = bytearray(input, 'utf-8')
        ecc = self.bch.encode(input_bytes)
        packet = input_bytes + ecc
        secret = [int(i) for i in ''.join(format(x, '08b') for x in packet)]
        assert len(secret) == 2 ** self.m, f"Encoded secret bits length should be {2 ** self.m}"
        return torch.Tensor(secret).type(torch.float32).unsqueeze(0)

    def decode_str(self, secrets: torch.Tensor, threshold: float = 0.5):
        n_errs, res = [], []
        for i in range(len(secrets)):
            bit_string = ''.join(str(int(k >= threshold)) for k in secrets[i])
            packet = self._bitstring_to_bytes(bit_string)
            data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
            n_err = self.bch.decode(data, ecc)
            if n_err < 0:
                n_errs.append(n_err)
                res.append([])
                continue
            self.bch.correct(data, ecc)
            packet = data + ecc
            try:
                n_errs.append(n_err)
                res.append(packet[:-self.bch.ecc_bytes].decode('utf-8'))
            except:
                n_errs.append(-1)
                res.append([])
        return n_errs, res

    def _bch_correct(self, secret: torch.Tensor, threshold: float = 0.5):
        bitstring = ''.join(str(int(x >= threshold)) for x in secret)
        packet = self._bitstring_to_bytes(bitstring)
        data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        n_err = self.bch.decode(data, ecc)
        if n_err < 0:
            logger.info("n_err < 0. Cannot accurately decode the message.")
            return packet
        self.bch.correct(data, ecc)
        return bytes(data + ecc)

    def _decode_data_bits(self, secrets: torch.Tensor, threshold: float = 0.5):
        return self.batch_decode_ecc(secrets, threshold)[:, :-self.bch.ecc_bytes * 8]

    def _bitstring_to_bytes(self, s):
        return bytearray(int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big'))
