# -*- coding: utf-8 -*-
"""
 This code tries to implement the MixMatch technique from the [paper](https://arxiv.org/pdf/1905.02249.pdf) MixMatch: A Holistic Approach to Semi-Supervised Learning and recreate their results on CIFAR10 with WideResnet28.

 It depends on Pytorch, Numpy and imgaug. The WideResnet28 model code is taken from [meliketoy](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py)'s github repository. Hopefully I can train this on Colab with a Tesla T4. :)
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mixmatch_utils import get_augmenter, mixmatch




#
#
#
#
# def to_torch(*args, device='cuda'):
#     convert_fn = lambda x: torch.from_numpy(x).to(device)
#     return list(map(convert_fn, args))
#
# """That about covers all the code we need for train and test loaders. Now we can start the training and evaluation. Let's see if all of this works or is just a mess. Going to add basically this same training code from meliketoy's repo but with the MixMatchLoss."""
#
# def test(model, test_gen, test_iters):
#     acc = []
#     for i, (x, y) in enumerate(test_gen):
#         x = to_torch(x)
#         pred = model(x).to('cpu').argmax(axis=1)
#         acc.append(np.mean(pred == y.argmax(axis=1)))
#         if i == test_iters:
#             break
#     print('Accuracy was : {}'.format(np.mean(acc)))
#
# def report(loss_history):
#     print('Average loss in last epoch was : {}'.format(np.mean(loss_history)))
#     return []
#
# def save(model, iter, train_iters):
#     torch.save(model.state_dict(), 'model_{}.pth'.format(train_iters // iters))
#
# def run(model, train_gen, test_gen, epochs, train_iters, test_iters, device):
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = MixMatchLoss()
#     loss_history = []
#     for i, (x, u, p, q) in enumerate(train_gen):
#         if i % train_iters == 0:
#             loss_history = report(loss_history)
#             test(model, test_gen, test_iters)
#             save(model, i, train_iters)
#             if i // train_iters == epochs:
#                 return
#         else:
#             optim.zero_grad()
#             x, u, p, q = to_torch(x, u, p, q, device=device)
#             loss = loss_fn(x, u, p, q, model)
#             loss.backward()
#             optim.step()
#             loss_history.append(loss.to('cpu'))
#
#
# import torch
# import torchvision
# import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



# training_amount + training_u_amount + validation_amount <= 50 000


def basic_generator(x, y=None, batch_size=32, shuffle=True):
    i = 0
    all_indices = np.arange(len(x))
    if shuffle:
        np.random.shuffle(all_indices)
    while(True):
        indices = all_indices[i:i+batch_size]
        if y is not None:
            yield x[indices], y[indices]
        else:
            yield x[indices]
        i = (i + batch_size) % len(x)

def mixmatch_wrapper(x, y, u, model, batch_size=32):
    augment_fn = get_augmenter()
    train_generator = basic_generator(x, y, batch_size)
    unlabeled_generator = basic_generator(u, batch_size=batch_size)
    while(True):
        xi, yi = next(train_generator)
        ui = next(unlabeled_generator)
        yield mixmatch(xi, yi, ui, model, augment_fn)

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg=img
    #npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


training_amount = 300

training_u_amount = 30000

validation_amount = 10000

transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

X_train = np.array(trainset.data)
y_train = np.array(trainset.targets)

X_test = np.array(testset.data)
y_test = np.array(testset.targets)

# Train set / Validation set split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_amount, random_state=1,
                                                          shuffle=True, stratify=y_train)

# Train unsupervised / Train supervised split
# Train set / Validation set split
X_train, X_u_train, y_train, y_u_train = train_test_split(X_train, y_train, test_size=training_u_amount, random_state=1,
                                                          shuffle=True, stratify=y_train)

X_remain, X_train, y_remain, y_train = train_test_split(X_train, y_train, test_size=training_amount, random_state=1,
                                                          shuffle=True, stratify=y_train)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = models.mobilenet_v2()  #TODO: Define "Wide ResNet-28"







# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

