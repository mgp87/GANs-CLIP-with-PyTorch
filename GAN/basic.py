import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def show(tensor, ch=1, size=(28,28), n=16):
    # tensor shape (128, 784)
    data = tensor.detach().cpu().view(-1, ch, *size)
    grid = make_grid(data[:n], nrow=n).permute(1,2,0)
    plt.imshow(grid)
    plt.show()

# Setup main params and hyperparams
epochs = 500
current_step = 0
show_step_interval = 300
mean_g_loss = 0 # mean of generator loss
mean_d_loss = 0 # mean of discriminator loss

z_dim = 64
learning_rate = 0.00001 # Usually used for Adam optimizer
loss_func = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss applying sigmoid function to the nn output

batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

# Number of steps = 60000 / 128 = 468

# Coding the generator loss
