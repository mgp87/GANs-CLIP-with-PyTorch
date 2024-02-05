## Generative AI for image generation: GANs & CLIP with PyTorch

This repo contains code from a course on Generative AI talking about GANs, CLIP with PyTorch. The course is available on Udemy by Javier Ideami.

### 1. Reinforcement Learning
- RL: Agents that learn to optimize target finding through trial and error by chasing rewards.
- In GenAI, the agent hallucinates scenarios where it tests different strategies to find the best one.

### 2. GAN Training
 ![GAN Training](https://github.com/mgp87/GANs-CLIP-with-PyTorch/blob/main/GAN/GAN_Training.png)

 #### 2.1. Cross Entropy Loss
    - The loss function used in GANs to measure the difference between the real and fake images by measuring the difference between the two probability distributions.

    *Info:* number of bits required to encode and transmit an event.
        **Lower probability** events have more info.
        **Higher probability** events have less info.
        $$\`h(x) = -log(P(x))\`$$

    **Entropy:** number of bits required to represent a randomly selected event from a probability distribution.
        **Skewed distribution** has **lower** entropy.
        **Uniform distribution** has **higher** entropy.
        $$`H(X) = -\sum_{i=1}^{n} P(x_i)log(P(x_i))`$$

    **Cross Entropy:** number of bits required to represent an event from one distribution using the probability distribution of another.
        - P = target distribution
        - Q = approximation of P
        - Cross-entropy is the number of extra bits needed to represent an event using Q instead of P.
        $$`H(P, Q) = -\sum_{i=1}^{n} P(x_i)log(Q(x_i))`$$
