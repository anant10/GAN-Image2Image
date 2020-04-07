from __future__ import absolute_import
import torch
from torch import nn
from torch.optim import Adam
from vanillaGAN import DiscriminatorNN
from vanillaGAN import GeneratorNN
from logger import Logger
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from util import TensorGenerator, ImageVectors
torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
out_dir = './dataset'
data = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)

discriminator = DiscriminatorNN()
generator = GeneratorNN()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

d_optimizer = Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = Adam(generator.parameters(), lr=0.0002)
loss = nn.BCELoss()


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, TensorGenerator.get_target_true(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, TensorGenerator.get_target_false(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, TensorGenerator.get_target_true(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


num_test_samples = 16
test_noise = TensorGenerator.get_random_vector(num_test_samples)


# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
device = torch.device("cuda")

num_epochs = 200
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(ImageVectors.get_vector(real_batch))
        real_data = real_data.to(device)

        # Generate fake data and detach
        # (so gradients are not calculated for generator)
        fake_data = generator(TensorGenerator.get_random_vector(N)).detach()
        fake_data = fake_data.to(device)
        # Train D
        d_error, d_pred_real, d_pred_fake =               train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(TensorGenerator.get_random_vector(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0:
            test_images = ImageVectors.get_image(generator(test_noise)).cpu()
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
        # Model Checkpoints
        logger.save_models(generator, discriminator, epoch)