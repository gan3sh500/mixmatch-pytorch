import torch
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


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
    
    
def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)
    
    
def mixup_mod(x1, x2, y1, y2, alpha):
    # lambda is a reserved word in python, substituting by beta
    beta = np.random.beta(alpha, alpha) 
    beta = np.amax([beta, 1-beta])
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y

def label_guessing(model, ub, K):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    probs = []
    for batch in ub:
        batch = torch.from_numpy(np.asarray(batch).transpose((0, 3, 1, 2))).to(device, dtype=torch.float)
        pr = model(batch) 
        probs.append(pr)

    sum = probs[0]
    for i in range(1,len(probs)):
        sum.add_(probs[i])

    return (sum/K).cpu().detach().numpy()

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Taken from Keras code
    https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def mixmatch(x, y, u, model, augment_fn, T=0.5, K=2, alpha=0.75):
    xb = augment_fn(x)
    y = to_categorical(y, num_classes=10) # Converting to one hot encode, num_clases=10 for future CIFAR test
    ub = [augment_fn(u) for _ in range(K)]
    avg_probs = label_guessing(model, ub, K)
    qb = sharpen(avg_probs, T)
    Ux = np.concatenate(ub, axis=0)
    Uy = np.concatenate([qb for _ in range(K)], axis=0)
    # Randon shuffle according to the paper
    indices = np.arange(len(xb) + len(Ux))
    np.random.shuffle(indices) 
    # MixUp
    Wx = np.concatenate([Ux, xb], axis=0)[indices]
    Wy = np.concatenate([Uy, y], axis=0)[indices]
    X, p = mixup_mod(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    U, q = mixup_mod(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
    
    # One hot decode for PyTorch labels compability
    p = p.argmax(axis=1)
    q = q.argmax(axis=1)
    return X, p, U, q


class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=100):
        self.lambda_u = lambda_u
        self.xent = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        super(MixMatchLoss, self).__init__()
    
    def forward(self, X, U, p, q):
        X_ = np.concatenate([X, U], axis=1)
        y_ = np.concatenate([p, q], axis=1)
        return self.xent(preds[:len(p)], p) + self.mse(preds[len(p):], q)
