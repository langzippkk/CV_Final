import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

from loader import *

dataset_train = PascalVOC(split = 'train')
dataset_val = PascalVOC(split = 'val')

NUM_TRAIN = 50
loader_train = DataLoader(dataset_train, batch_size=2,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
 
# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it heremeouterrorhttps://accounts.google.com/o/oauth2/approval/v2/approvalnativeapp?auto=false&response=code%3D4%2FxQFcInhFnNrjjDafD8vcWhDAVFJsXNLV-629N1FbixYxTvfpHRBNkT8%26scope%3Demail%2520https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email%2520https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%2520https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%2520https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%2520https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly%2520openid%26authuser%3D0%26prompt%3Dconsent&hl=en&approvalCode=4%2FxQFcInhFnNrjjDafD8vcWhDAVFJsXNLV-629N1FbixYxTvfpHRBNkT8
backbone.out_channels = 1280
 
# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and/data/
# aspect ratios 
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),        
))
 
# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)
# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

model.parameters

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


def train(n_epoch, model, train_loader, optimizer, epoch, freq):
    count = 0

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):

        """
        TODO: Implement training loop.
        """
        model.train()
        optimizer.zero_grad()
        print("before")
        outputs = model(images.float(), labels)
        print("after")

        loss = cross_entropy2d(outputs, labels, freq)
        loss.backward()
        optimizer.step()
        # raise NotImplementedError

        if batch_idx % 20 == 0:
            count = count + 1
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, n_epoch, loss.data))

        # if batch_idx % 20 == 0:
        #     """
        #     Visualization of results.
        #     """
        #     pred = outputs[0,:,:,:]
        #     gt = labels[0,:,:].data.numpy().squeeze()
        #     im = images[0,:,:,:].data.numpy().squeeze()
        #     im = np.swapaxes(im, 0, 2)
        #     im = np.swapaxes(im, 0, 1)
        #     _, pred_mx = torch.max(pred, 0)        train(args, zoomout, classifier, train_loader, optimizer, epoch, freq)

            
        #     # pred_mx = torch.abs((pred*19).int())
        #     # print(pred_mx.size())

        #     pred = pred_mx.data.numpy().squeeze()
        #     image = Image.fromarray(im.astype(np.uint8), mode='RGB')

        #     image.save("./imgs/im_" + str(count) + "_" + str(epoch) + "_.png")
        #     visualize("./lbls/pred_" + str(count) + "_" + str(epoch) + ".png", pred)
        #     visualize("./lbls/gt_" + str(count) + "_" + str(epoch) + ".png", gt)

    # Make sure to save your model periodically
    torch.save(model, "./models/full_model.pkl")

num_classes = 2

# get the model using our helper function
# move model to the right device
model.to(device)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations

    # train_one_epoch(model, optimizer, loader_train, device, epoch, print_freq=0)
    # update the learning rate
    train(num_epochs, model, loader_train, optimizer, epoch, freq=0)
    lr_scheduler.step()

    # evaluate on the test daheadertaset
