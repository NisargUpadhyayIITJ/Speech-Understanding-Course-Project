import torch

from model import aasist3


model = aasist3()
model.eval()
model.to("cuda:5")

batch_size = 2
num_samples = 64600
mock_audio_tensor = torch.randn(batch_size, num_samples)

print(mock_audio_tensor.shape)

with torch.no_grad():
    output = model(mock_audio_tensor.to("cuda:5"))
        