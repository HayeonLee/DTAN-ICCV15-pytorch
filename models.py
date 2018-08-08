import torch
import torch.nn as nn

class DTAN(nn.Module):
	def __init__(self, img_size, in_ch, out_ch, num_block=2):
		super(DTAN, self).__init__()
		layers = []
		for i in range(num_block):
			layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1)) #bias?
			layers.append(nn.BatchNorm2d(out_ch)) #Replace LRU -> BatchNorm2d
			layers.append(nn.ReLU(inplace=True))
			layers.append(nn.Dropout2d(p=0.2))
			layers.append(nn.MaxPool2d(2,2)) #MaxPool2d vs strided conv
			in_ch = out_ch
		self.cnns = nn.Sequential(*layers)
		self.fc1 = nn.Linear(pow((img_size / pow(2, num_block)), 2) * out_ch, 500)
		self.fc2 = nn.Linear(500, 7)

	def forward(self, x):
		x = self.cnns(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		out = self.fc2(x)
		return out
