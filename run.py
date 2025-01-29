import torch
from torchvision import transforms
from PIL import Image

from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.fhe import Configuration

from functools import reduce
import bitarray

import train
import utils
import noise


# 1. Let us implement a simple Hamming error correction code
# You can change this to a more sophisticated and efficient error correction algorithm.
# Here is just a simple example.
def tensor2bitarray(t):
    listed_secret = []
    for i in range(len(t)):
        listed_secret.append(int(t[i].item()))
    return bitarray.bitarray(listed_secret)


def bitarray2tensor(b):
    return torch.tensor([int(x) for x in b])


def hamming_encode(bits):
    bits = bits[:]
    bits.insert(0, 0)
    [bits.insert(2 ** x, 1) for x in range(4)]
    xor = reduce(lambda x, y: x ^ y, [i for i, bit in enumerate(bits) if int(bit) == 1])
    negate = [2 ** i for i, bit in enumerate(bin(xor)[:1:-1]) if int(bit) == 1]
    [bits.invert(x) for x in negate]
    if bits.count(1) % 2 == 1:
        bits[0] = 1

    return bits

def hamming_decode(bits):
    # detect error, fix error and return the original bits
    bits = bits[:]
    xor = reduce(lambda x, y: x ^ y, [i for i, bit in enumerate(bits) if int(bit) == 1])
    if xor != 0:
        bits.invert(xor)

    return [bit for i, bit in enumerate(bits) if i not in [0, 1, 2, 4, 8]]


# 2. Load the model
cfg = torch.load('./ckpts/cfg.pth')
wm_model = train.Watermark(cfg, device='cpu')
encoder = wm_model.encoder
decoder = wm_model.decoder
encoder.load_state_dict(torch.load('./ckpts/encoder.pth'))
decoder.load_state_dict(torch.load('./ckpts/decoder.pth'))

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 3. Compiling the model
# prepare the input and the secret
input = transform(Image.open('./example.jpg').convert('RGB')).unsqueeze(0)
# randomly generate a secret
secret, _ = utils.uuid_to_bits(1)
# the model is pretrained using 16 bits secrets
original_secret = secret[:, :11]
secret_bits = tensor2bitarray(original_secret[0])
hamming_secret = hamming_encode(secret_bits)
secret = bitarray2tensor(hamming_secret).unsqueeze(0).float()
print("secret:", original_secret)
print("secret with Hamming ECC:", secret)
# compiling
config = Configuration(
    enable_tlu_fusing=True,
    print_tlu_fusing=False,
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache",
    show_progress=True,
    use_gpu=False
)
quant_encoder = compile_brevitas_qat_model(
    encoder,
    (input, secret),
    rounding_threshold_bits={"n_bits": 7, "method": "approximate"},
    configuration=config,
    verbose=True
)
print("maximum_integer_bit_width: ", quant_encoder.fhe_circuit.graph.maximum_integer_bit_width())
print("statistics: ", quant_encoder.fhe_circuit.statistics)

# 4. Evaluation
noiser = noise.Noiser(num_transforms=1, device="cpu")
encoded_input = quant_encoder.forward(input.numpy(), secret.numpy(), fhe="execute")
noised_input = noiser(torch.from_numpy(encoded_input).float(), ["RandomResizedCrop", "Jiggle"])
noised_decoded_secret = decoder(noised_input) > 0.5
noised_secret_bits = tensor2bitarray(noised_decoded_secret[0])
noised_secret = hamming_decode(noised_secret_bits)
noised_secret = bitarray2tensor(noised_secret).unsqueeze(0).float()
print("Original secret", original_secret)
print("Decoded secret", noised_secret)
print("Secret recovered? ", torch.all(noised_secret == original_secret).item())
