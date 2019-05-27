# -*- coding: utf-8 -*-
"""
 This code tries to implement the MixMatch technique from the [paper](https://arxiv.org/pdf/1905.02249.pdf) MixMatch: A Holistic Approach to Semi-Supervised Learning and recreate their results on CIFAR10 with WideResnet28.

 It depends on Pytorch, Numpy and imgaug. The WideResnet28 model code is taken from [meliketoy](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py)'s github repository. Hopefully I can train this on Colab with a Tesla T4. :)
"""



import torch
import numpy as np
import imgaug.augmenters as iaa

"""Now that we have the basic imports out of the way lets get to it. 
First we shall define the function to get augmented version of a given batch of images. The below function returns the function to do that.
"""

def get_augmenter():
    seq = iaa.Sequential([iaa.Fliplr(0.5), # horrizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
    ])
    def augment(images):
        # Only works with list. Convert np to list
        imgs = []
        for i in range(images.shape[0]):
            imgs.append(images[i,:,:,:])

        images = images

        return seq.augment(images=images)
    return augment

"""Next we define the sharpening function to sharpen the prediction from the averaged prediction of all the unlabeled augmented images. It does the same thing as applying a temperature within the softmax function but to the probabilities."""

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)

"""A simple implementation of the [paper](https://arxiv.org/pdf/1710.09412.pdf) mixup: Beyond Empirical Risk Minimization used in this paper as well."""

def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, -alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y

"""This covers Algorithm 1 from the paper."""

def mixmatch(x, y, u, model, augment_fn, T=0.5, K=2, alpha=0.75):
    xb = augment_fn(x)
    ub = [augment_fn(u) for _ in range(K)]
    qb = sharpen(sum(map(lambda i: model(i), ub)) / K, T)
    Ux = np.concatenate(ub, axis=0)
    Uy = np.concatenate([qb for _ in range(K)], axis=0)
    indices = np.random.shuffle(np.arange(len(xb) + len(Ux)))
    Wx = np.concatenate([Ux, xb], axis=0)[indices]
    Wy = np.concatenate([qb, y], axis=0)[indices]
    X, p = mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    U, q = mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
    return X, U, p, q

"""The combined loss for training from the paper."""

class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=100):
        self.lambda_u = lambda_u
        self.xent = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        super(MixMatchLoss, self).__init__()
    
    def forward(self, X, U, p, q, model):
        X_ = np.concatenate([X, U], axis=1)
        preds = model(X_)
        return self.xent(preds[:len(p)], p) + \
                                    self.lambda_u * self.mse(preds[len(p):], q)

"""Now that we have the MixMatch stuff done, we have a few things to do. Namely, define the WideResnet28 model, write the data and training code and write testing code. 
Let's start with the model. The below is just a copy paste mostly from the wide-resnet.pytorch repo by meliketoy.
"""

def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           bias=True)

"""Will need the below init function later before training."""

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        torch.nn.init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant(m.weight, 1)
        torch.nn.init.constant(m.bias, 0)

"""The basic block for the WideResnet"""

class WideBasic(torch.nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3,
                                     padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                                     padding=1, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, planes, kernel_size=1,
                                stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(torch.nn.functional.relu(self.bn1(x))))
        out = self.conv2(torch.nn.functional.relu(self.bn2(out)))
        return out + self.shortcut(x)

"""Aaand the full model with default params set for CIFAR10."""

class WideResNet(torch.nn.Module):
    def __init__(self, depth=28, widen_factor=10,
                 dropout_rate=0.3, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self.wide_layer(WideBasic, nStages[1], n, dropout_rate,
                                      stride=1)
        self.layer2 = self.wide_layer(WideBasic, nStages[2], n, dropout_rate,
                                      stride=2)
        self.layer3 = self.wide_layer(WideBasic, nStages[3], n, dropout_rate,
                                      stride=2)
        self.b1 = torch.nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = torch.nn.Linear(nStages[3], num_classes)
    
    def wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer3(self.layer2(self.layer1(out)))
        out = torch.nn.functional.relu(self.bn1(out))
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)

"""Now that we have the model let's write train and test loaders so that we can pass the model and the data to the MixMatchLoss."""

def basic_generator(x, y=None, batch_size=32, shuffle=True):
    i = 0
    all_indices = np.random.shuffle(np.arange(len(x))) if shuffle else \
                                                               np.arange(len(x))
    while(True):
        indices = all_indices[i:i+batch_size]
        if y is not None:
            yield x[indices], y[indices]
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

def to_torch(*args, device='cuda'):
    convert_fn = lambda x: torch.from_numpy(x).to(device)
    return list(map(convert_fn, args))

"""That about covers all the code we need for train and test loaders. Now we can start the training and evaluation. Let's see if all of this works or is just a mess. Going to add basically this same training code from meliketoy's repo but with the MixMatchLoss."""

def test(model, test_gen, test_iters):
    acc = []
    for i, (x, y) in enumerate(test_gen):
        x = to_torch(x)
        pred = model(x).to('cpu').argmax(axis=1)
        acc.append(np.mean(pred == y.argmax(axis=1)))
        if i == test_iters:
            break
    print('Accuracy was : {}'.format(np.mean(acc)))

def report(loss_history):
    print('Average loss in last epoch was : {}'.format(np.mean(loss_history)))
    return []

def save(model, iter, train_iters):
    torch.save(model.state_dict(), 'model_{}.pth'.format(train_iters // iters))

def run(model, train_gen, test_gen, epochs, train_iters, test_iters, device):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MixMatchLoss()
    loss_history = []
    for i, (x, u, p, q) in enumerate(train_gen):
        if i % train_iters == 0:
            loss_history = report(loss_history)
            test(model, test_gen, test_iters)
            save(model, i, train_iters)
            if i // train_iters == epochs:
                return
        else:
            optim.zero_grad()
            x, u, p, q = to_torch(x, u, p, q, device=device)
            loss = loss_fn(x, u, p, q, model)
            loss.backward()
            optim.step()
            loss_history.append(loss.to('cpu'))