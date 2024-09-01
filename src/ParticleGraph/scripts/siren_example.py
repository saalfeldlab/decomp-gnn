# Example of SIREN network
# Following https://github.com/vsitzmann/siren?tab=readme-ov-file

import torch
from torch import nn
from torch.utils.data import Dataset

from tqdm import trange
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import numpy as np
import skimage
import matplotlib.pyplot as plt
import imageio
# from torchinfo import summary

from ParticleGraph.models import Siren_Network



def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def load_image(path, crop_width=None, device='cpu'):
    target = imageio.imread(path).astype(np.float16)
    target = target / np.max(target)
    target = torch.tensor(target).unsqueeze(0).to(device)
    if crop_width is not None:
        target = CenterCrop(crop_width)(target)
    return target


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    image_width = 256

    # Example image (cameraman)
    # target = get_cameraman_tensor(image_width)

    # Custom image file
    image_file = '../../../graphs_data/pattern_1.tif' # boat_512.tif, beads_abb.tif, beads_gt.tif
    target = load_image(image_file, crop_width=image_width, device=device)


    # Train SIREN model
    model = Siren_Network(image_width=image_width, in_features=2, out_features=1, hidden_features=128, hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80, hidden_omega_0=80.)
    # NOTE: higher omega can represent higher frequency signals

    total_steps = 1000
    steps_til_summary = 200

    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

    model_input = get_mgrid(image_width, 2)
    ground_truth = target.reshape(-1, 1)
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    model=model.to(device)

    for step in trange(total_steps+1):
        model_output, coords = model()
        loss = ((model_output - ground_truth)**2).mean()
        
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1,4, figsize=(28,5))
            axes[0].imshow(target.cpu().squeeze().numpy())
            axes[0].set_title('Original')
            axes[1].imshow(model_output.cpu().view(image_width,image_width).detach().numpy())
            axes[1].set_title('Generated')
            axes[2].imshow(img_grad[:,1].cpu().view(image_width,image_width).detach().numpy())
            axes[2].set_title('Gradient_x')
            axes[3].imshow(img_grad[:,0].cpu().view(image_width,image_width).detach().numpy())
            axes[3].set_title('Gradient_y')
            plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()