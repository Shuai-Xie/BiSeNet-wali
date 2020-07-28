import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torchvision.models.resnet import model_urls


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,  # 只有1次
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, inplanes, layers, output_stride, BatchNorm, pretrained=True):
        super(ResNet, self).__init__()  # inplanes 默认 =64, 可调节小网络
        blocks = [1, 2, 4]  # multi grids
        self.inplanes = inplanes

        # before layers, out 1/4
        # layer3 2/1 for different output stride
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]  # 1/4, 1/8, 1/16, 1/16
        elif output_stride == 8:  # strides 少1个2; layer3,4, dilation x2
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1/4

        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=strides[0], dilation=dilations[0],  # 1/4
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=strides[1], dilation=dilations[1],  # 1/8
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=strides[2], dilation=dilations[2],  # 1/16
                                       BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
        #                                BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, inplanes * 8, blocks=blocks, stride=strides[3], dilation=dilations[3],  # 1,2
                                         BatchNorm=BatchNorm)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model(layers)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        """
        :param block: BasicBlock, Bottleneck
        :param planes: features num = planes * block.expansion
        :param blocks: block repeat times
        :param stride: 1st conv's stride of current layer
        :param dilation:
        :param BatchNorm:
        :return:
        """
        # layer 连接处，首层残差连接 是否需要 downsample
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        # 首个 block
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        # 内部 block
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        """
        级联 dilation 模块，参考 deeplabv3+，维持 1/16, 但采集更大感受野 feature
        blocks: [1, 2, 4]
        stride=1, dilation=2
        layer dilations: 2,4,8
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 1/4
        x = self.layer2(x)  # 1/8
        x3 = self.layer3(x)  # 1/16
        x4 = self.layer4(x3)  # 1/16
        return x3, x4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, layers):
        if layers == [3, 4, 23, 3]:
            pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        elif layers == [3, 4, 6, 3]:
            pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        elif layers == [2, 2, 2, 2]:
            pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
        else:
            raise NotImplementedError
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # 参数 name & 参数 size 双重判断
            if k in state_dict and v.size() == state_dict[k].size():
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def build_contextpath(model, inplanes=64, output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True):
    if model == 'resnet18':
        return ResNet(BasicBlock, inplanes, [2, 2, 2, 2], output_stride, BatchNorm, pretrained=pretrained)
    elif model == 'resnet50':
        return ResNet(Bottleneck, inplanes, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
    elif model == 'resnet101':
        return ResNet(Bottleneck, inplanes, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)


if __name__ == "__main__":
    import torch

    model = build_contextpath('resnet50', inplanes=32, pretrained=True)
    input = torch.rand(1, 3, 512, 512)
    feature3, feature4 = model(input)
    print(feature3.size())  # 1/8
    print(feature4.size())  # 1/4
