## Generative AI for image generation: GANs & CLIP with PyTorch

This repo contains code from a course on Generative AI talking about GANs, CLIP with PyTorch.

### 1. Reinforcement Learning
- RL: Agents that learn to optimize target finding through trial and error by chasing rewards.
- In GenAI, the agent hallucinates scenarios where it tests different strategies to find the best one.

### 2. GAN Training
GAN stands for Generative Adversarial Networks. It is a type of neural network that is used to generate new data that is similar to the training data. It is composed of two networks: the generator and the discriminator.

 ![GAN Training](https://github.com/mgp87/GANs-CLIP-with-PyTorch/blob/main/GAN/GAN_Training.png)

 #### 2.1. Cross Entropy Loss
- The loss function used in GANs to measure the difference between the real and fake images by measuring the difference between the two probability distributions.

*Info:* number of bits required to encode and transmit an event.
**Lower probability** events have more info.
**Higher probability** events have less info.

```math
h(x) = -log(P(x))
```

**Entropy:** number of bits required to represent a randomly selected event from a probability distribution.
**Skewed distribution** has **lower** entropy.
**Uniform distribution** has **higher** entropy.

```math
H(X) = -\sum_{i=1}^{n} P(x_i)log(P(x_i))
```

**Cross Entropy:** number of bits required to represent an event from one distribution using the probability distribution of another.
- P = target distribution
- Q = approximation of P
- Cross-entropy is the number of extra bits needed to represent an event using Q instead of P.

```math
H(P, Q) = -\sum_{i=1}^{n} P(x_i)log(Q(x_i))
```

#### 2.2. Discriminator Loss

```math
BCELoss = -1/n\sum_{i=1}^{n} [y_ilog(\hat{y_i}) + (1 - y_i)log(1 - \hat{y_i})]
```

**When label is 1 (real):**
```math
log(\hat{y_i})
```

**When label is 0 (fake):**
```math
log(1 - \hat{y_i})
```

**Combined:**
```math
-1/n\sum_{i=1}^{n} (logD({x_i}) + log(1-D(G(z^{i}))))
```

**MinMax Game:** The generator tries to minimize the loss, while the discriminator tries to maximize it.

```math
\min_d -[E(logD(x)) + E(log(1-D(G(z))))]
```

#### 2.3. Generator Loss

```math
BCELoss = -1/n\sum_{i=1}^{n} [y_ilog(\hat{y_i}) + (1 - y_i)log(1 - \hat{y_i})]
```

**When label is 1 (real):**
```math
log(\hat{y_i})
```

```math
-1/n\sum_{i=1}^{n} log(D(G({z_i})))
```

```math
\min_g -[E(log(D(G(z))))]
```

- [Basic GAN python script](https://github.com/mgp87/GANs-CLIP-with-PyTorch/blob/main/GAN/basic.py)
- [Basic GAN notebook](https://github.com/mgp87/GANs-CLIP-with-PyTorch/blob/main/GAN/basic.ipynb)

The BCELoss function has the problem of ***Mode Collapse***, where the generator produces the same output for all inputs. This is because the generator is trying to minimize the loss, and the discriminator is trying to maximize it, so the generator ends up stuck in a single mode (peak of distribution).

We can find the ***Flat Gradient*** problem, where the gradients of the generator loss are flat, so the generator is not able to learn from the gradients.

#### 2.4. Wasserstein GAN (WGAN)

The WGAN uses the Wasserstein distance (also known as Earth Mover's distance) to measure the difference between the real and fake images.

##### 2.4.1. Wasserstein Loss

```math
-1/n\sum_{i=1}^{n} (hat{y_i}{pred_i})
```

**Critic loss:**
Critic loss will try to maximize the difference between the real and fake images.

```math
min_d -[E(D(x)) - E(D(G(z)))]
```

**Generator loss:**
Generator loss will try to minimize the difference between the real and fake images.

```math
min_g -[E(D(G(z)))]
```

We will have the ***MinMax Game*** again, where the generator tries to minimize the loss, while the discriminator tries to maximize it.

- Wloss helps with mode collapse and vanishing gradient issues becoming more stable than BCELoss.
- WGAN is more stable and has better convergence properties than the original GAN.

###### 2.4.1.1. Gradient Penalty

The gradient penalty is used to enforce the Lipschitz constraint, which is a condition that ensures the gradients of the discriminator are not too large.

Lipschitz continuous condition is a condition that ensures the gradients of the discriminator are not too large. The norm of the gradients of the discriminator should be 1 or less than 1.

This is mandatory for a stable training process when using WLoss ensuring to approximate Earth Mover's distance the best way possible.

**Condition application:**
1. Weight clipping: Clip the weights of the discriminator to enforce the Lipschitz constraint after each update (interferes with the learning process of the critic).

2. Gradient penalty: Add a penalty term to the loss function that enforces the Lipschitz constraint without interfering with the learning process of the critic. Regularization term is added to the loss function to ensure the critic satisfies the Lipschitz constraint (1-L continuous). This is the preferred method.

```math
min_g max_c [E(c(x)) - E(c(z))] + \lambda gp
```
where:

```math
gp = (||\bigtriangledown c(x)||_2 - 1)^2
```

```math
x = \alpha * real + (1 - \alpha)*fake
```

x is an interpolation between the real and fake images.

Substituing gp in the loss function:

```math
min_g max_c [E(c(x)) - E(c(z))] + \lambda E(||\bigtriangledown c(x)||_2 - 1)^2
```

###### 2.4.1.2. Convolutions

The convolutional layers are used to extract features from the images, and the fully connected layers are used to classify the images.

***nn.Conv2d:*** Applies a 2D (width and height) convolution over an input signal composed of several input planes. Typically reduces the size of the input. It will be used in the critic.

```math
new_width = (old_width + 2*padding - dilation x (kernel_size - 1) // stride + 1
```

```math
Simplified: (n+2*pad - ks)//stride + 1
```

***nn.ConvTranspose2d:*** Applies a 2D transposed convolution operator over an input image composed of several input planes. Typically increases the size of the input. It is almost the opposite of the Conv2d. It will be used in the generator.

```math
new_width = (old_width - 1) x stride -2 x pad + dilation x (kernel_size - 1) + output_pad + 1
```

```math
Simplified: (n-1)*stride - 2*pad + ks
```

- pad (padding): The number of pixels to add to each side of the input.
- stride: The number of pixels to move the kernel at each step.
- ks (kernel size): The size of the kernel, 3 for a 3x3 kernel.
- dilation: The number of pixels to skip in the kernel.
- n: the old width or height of the image.

Convolutions are really good at detecting patterns in images, and they are also good at detecting patterns in text --> Translation invariant way: the same pattern can be detected in different parts of the image.

Convolution is a grid of numbers that are going to be initialized randomly, and then they are going to be learned during the training process. By positioning the grid of numbers (kernel) in different parts of the image multiplying the values of the kernel by the values of the image, and then summing the results, which is the output of the convolution operation. Depending on the value of the image, it will detect different parts of the image. The deeper the network, the more complex the patterns it can detect.

As you apply convolutions, the image size will become smaller. In order to compensate for it, we increase the number of channels
