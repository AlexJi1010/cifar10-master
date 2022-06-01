import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from network import Network
from vgg16modified import Network as MVGG16Net
from utils import device, get_all_preds, get_num_correct
import matplotlib.pyplot as plt
import numpy as np

def run():
    torch.multiprocessing.freeze_support()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_set = transform
    # test_set(img)
    test_set = torchvision.datasets.CIFAR10(
        root='./data/',
        train=False,
        download=False,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, num_workers=1)

    # load the model with least validation loss
    model = MVGG16Net(torchvision.models.vgg16(pretrained=True)).to(device)
    model.load_state_dict(
        torch.load('models/model-transferlr_vgg16.pth',
                   map_location=device)
    )
    model
    model.eval()
    # obtain one batch of test images
    images, labels = next(iter(test_loader))
    images = images.numpy()
    test_preds = get_all_preds(model, test_loader)
    test_stacked = torch.stack(
        (torch.as_tensor(test_set.targets), test_preds.argmax(dim=1)),
        dim=1
    )
    # plot the images in the batch, along with predicted and true labels
    # plot format: "predicted-label (true-label)"
    fig = plt.figure(figsize=(25, 20))
    for i in np.arange(100):
        ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        image = np.squeeze(np.transpose(images[i], (1, 2, 0)))  # B*C*H*W --> H*W*C
        image = image * np.array((0.229, 0.224, 0.225)) + \
                np.array((0.485, 0.456, 0.406))  # un-normalize the image
        image = image.clip(0, 1)  # clip the values between 0 and 1
        ax.imshow(image)
        ax.set_title(f'{test_set.classes[test_stacked[i, 1].item()]} ({test_set.classes[labels[i].item()]})',
                     color=('green' if test_stacked[i, 1] == labels[i] else 'red'))

    plt.show()
    fig.savefig('visualizations/testest.png', bbox_inches='tight')
    plt.close()
    print('done')

if __name__ == '__main__':
    run()

# img = Image.open("./data/img.png")
