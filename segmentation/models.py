import torch 
import torch.nn.functional as F

def conv3x3_bn(ci, co):
  return torch.nn.Sequential(torch.nn.Conv2d(ci, co, 3, padding=1), torch.nn.BatchNorm2d(co), torch.nn.ReLU(inplace=True))

def encoder_conv(ci, co):
  return torch.nn.Sequential(torch.nn.MaxPool2d(2), conv3x3_bn(ci, co), conv3x3_bn(co, co))

class deconv(torch.nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
    self.conv1 = conv3x3_bn(ci, co)
    self.conv2 = conv3x3_bn(co, co)

  def forward(self, x1, x2):
    x1 = self.upsample(x1)
    diffX = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (diffX, 0, diffY, 0))
    # concatenating tensors
    x = torch.cat([x2, x1], dim=1)
    x = self.conv1(x)
    x = self.conv2(x)
    return x

class UNet(torch.nn.Module):
  def __init__(self, n_classes=1, in_ch=3):
    super().__init__()

    # number of filter's list for each expanding and respecting contracting layer
    c = [16, 32, 64, 128]

    # first convolution layer receiving the image
    self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
                                     conv3x3_bn(c[0], c[0]))

    # encoder layers
    self.conv2 = encoder_conv(c[0], c[1])
    self.conv3 = encoder_conv(c[1], c[2])
    self.conv4 = encoder_conv(c[2], c[3])

    # decoder layers
    self.deconv1 = deconv(c[3],c[2])
    self.deconv2 = deconv(c[2],c[1])
    self.deconv3 = deconv(c[1],c[0])

    # last layer returning the output
    self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

  def forward(self, x):
    # encoder
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x = self.conv4(x3)
    # decoder
    x = self.deconv1(x, x3)
    x = self.deconv2(x, x2)
    x = self.deconv3(x, x1)
    x = self.out(x)
    return x