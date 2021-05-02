import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        # atten = F.avg_pool2d(feat, feat.size()[2:])
        # atten = F.adaptive_avg_pool2d(feat, (1,1))
        atten = torch.mean(feat, (2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        # H8, W8 = feat8.size()[2:]
        # H16, W16 = feat16.size()[2:]
        # H32, W32 = feat32.size()[2:]

        # avg = F.avg_pool2d(feat32, feat32.size()[2:])
        # avg = F.adaptive_avg_pool2d(feat32, (1, 1))
        avg = torch.mean(feat32, (2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=(int(feat32.shape[2]), int(feat32.shape[3])), mode='nearest')

        feat32_arm = self.arm32(feat32)
        # feat32_sum = feat32_arm + avg_up
        feat32_sum = torch.add(feat32_arm, avg_up)
        feat32_up = F.interpolate(feat32_sum, size=(int(feat16.shape[2]), int(feat16.shape[3])), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        # feat16_sum = feat16_arm + feat32_up
        feat16_sum = torch.add(feat16_arm, feat32_up)
        feat16_up = F.interpolate(feat16_sum, size=(int(feat8.shape[2]), int(feat8.shape[3])), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        return feat8, feat16_up  # x8, x8

### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        # atten = F.avg_pool2d(feat, feat.size()[2:])
        # atten = F.adaptive_avg_pool2d(feat, (1, 1))
        atten = torch.mean(feat, (2,3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        # feat_out = feat_atten + feat
        feat_out = torch.add(feat_atten,feat)
        return feat_out

class BiSeNet(nn.Module):
    def __init__(self, n_classes):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)

    def forward(self, x):
        feat_res8, feat_cp8 = self.cp(x)  # here return res3b1 feature
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out = F.interpolate(feat_out, size=(int(x.shape[2]), int(x.shape[3])), mode='bilinear', align_corners=True)
        return feat_out

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = BiSeNet(19).to(device)
    net.eval()

    # own_state = net.state_dict()
    # model = torch.load('79999_iter.pth', map_location=device)
    # params = []
    # for name in model:
    #     if not (name.startswith('conv_out32.') or name.startswith('conv_out16.')):
    #         params.append((name, model[name]))
    # print(len(own_state), len(params))
    # for i, name in enumerate(own_state.keys()):
    #     own_state[name].copy_(params[i][1])
    # torch.save(own_state, 'my_params.pth')

    with torch.no_grad():
        in_ten = torch.randn(2, 3, 512, 512).to(device)
        out = net(in_ten)
        print(out.shape)

    inputs = torch.randn(1, 3, 512, 512).to(device)
    torch.onnx.export(net, inputs, 'my_param.onnx', verbose=False, opset_version=12, input_names=["input"], output_names=["out"])
