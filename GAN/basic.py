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
    grid = make_grid(data[:n], nrow=4).permute(1,2,0)
    plt.imshow(grid)
    plt.show()

# Setup main params and hyperparams
epochs = 500
current_step = 0
show_step_interval = 300
mean_g_loss = 0 # mean of generator loss
mean_d_loss = 0 # mean of discriminator loss

d_z = 64
learning_rate = 0.00001 # Usually used for Adam optimizer
loss_func = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss applying sigmoid function to the nn output

batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

# Number of steps = 60000 / 128 = 468

# Generator Block
def generator_block(input, output):
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, d_z=64, d_i=784, d_h=128):
        super().__init__()
        self.gen = nn.Sequential(
            generator_block(d_z, d_h), # 64, 128
            generator_block(d_h, d_h*2), # 128, 256
            generator_block(d_h*2, d_h*4), # 256, 512
            generator_block(d_h*4, d_h*8), # 512, 1024
            nn.Linear(d_h*8, d_i), # 1024, 784 --> 28x28
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

def generator_noise(n, d_z):
    return torch.randn(n, d_z).to(device)

# Discriminator Block
def discriminator_block(input, output):
    return nn.Sequential(
        nn.Linear(input, output),
        nn.LeakyReLU(0.2, inplace=True) # LeakyReLU is used to prevent dying ReLU problem
    )

class Discriminator(nn.Module):
    def __init__(self, d_i=784, d_h=256):
        super().__init__()
        self.disc = nn.Sequential(
            discriminator_block(d_i, d_h*4), # 784, 1024
            discriminator_block(d_h*4, d_h*2), # 1024, 512
            discriminator_block(d_h*2, d_h), # 512, 256
            nn.Linear(d_h, 1) # 256, 1
        )

    def forward(self, image):
        return self.disc(image)

# Generator Loss
def generator_loss(loss_func, gen, disc, n, d_z):
    noise = generator_noise(n, d_z)
    fake = gen(noise)
    pred = disc(fake)
    targets = torch.ones_like(pred)
    g_loss = loss_func(pred, targets)
    return g_loss

# Discriminator Loss
def discriminator_loss(loss_func, gen, disc, real, n, d_z):
    noise = generator_noise(n, d_z)
    fake = gen(noise)
    disc_fake = disc(real.detach()) # detach to prevent backpropagation to generator
    fake_targets = torch.zeros_like(disc_fake)
    disc_fake_loss = loss_func(disc_fake, fake_targets)
    disc_real = disc(real)
    real_targets = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, real_targets)
    disc_loss = (disc_fake_loss + disc_real_loss)
    return disc_loss

# Training Loop
# Each step will process 128 images = size of the batch

def training_loop(gen, disc, generator_optimizer, discriminator_optimizer, loss_func, dataloader, epochs, d_z):
    for epoch in range(epochs):
        for real, _ in tqdm(dataloader):
            # Train Discriminator
            discriminator_optimizer.zero_grad()
            current_batch_size = len(real) # len(real) = 128 x 1 x 28 x 28
            real = real.view(current_batch_size, -1).to(device)  # 128 x 784
            disc_loss = discriminator_loss(loss_func, gen, disc, real, current_batch_size, d_z)
            disc_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()
            gen_loss = generator_loss(loss_func, gen, disc, current_batch_size, d_z)
            gen_loss.backward(retain_graph=True)
            generator_optimizer.step()

            # Visualization
            mean_disc_loss += disc_loss.item() / show_step_interval
            mean_gen_loss += gen_loss.item() / show_step_interval

            if current_step % show_step_interval == 0:
                fake_noise = generator_noise(current_batch_size, d_z)
                fake = gen(fake_noise)
                show(fake)
                show(real)
                print(f"Epoch {epoch}, step {current_step}: Generator loss: {mean_gen_loss}, Discriminator loss: {mean_disc_loss}")


if __name__ == '__main__':
    gen = Generator(d_z).to(device)
    generator_optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate)

    disc = Discriminator().to(device)
    discriminator_optimizer = torch.optim.Adam(disc.parameters(), lr=learning_rate)

    print(gen)
    print(disc)

    x, y = next(iter(dataloader))
    print(x.shape, y.shape)
    print(y[:10])

    noise = generator_noise(batch_size, d_z)
    fake = gen(noise)
    show(fake)

    training_loop(gen, disc, generator_optimizer, discriminator_optimizer, loss_func, dataloader, epochs, d_z)
