import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # return out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # return x
        return self.sigmoid(x)


class CoAttention(nn.Module):
    def __init__(self, channel):
        super(CoAttention, self).__init__()

        d = channel // 16
        self.proja = nn.Conv2d(channel, d, kernel_size=1)
        self.projb = nn.Conv2d(channel, d, kernel_size=1)

        self.bottolneck1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.bottolneck2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.proj1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.proj2 = nn.Conv2d(channel, 1, kernel_size=1)

        self.bna = nn.BatchNorm2d(channel)
        self.bnb = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, Qa, Qb):
        # cascade 1
        Qa_1, Qb_1 = self.forward_sa(Qa, Qb)
        _, Zb = self.forward_co(Qa_1, Qb_1)

        Pa = F.relu(Zb + Qa)
        Pb = F.relu(Qb_1 + Qb)

        # cascade 2
        Qa_2, Qb_2 = self.forward_sa(Pa, Pb)
        _, Zb = self.forward_co(Qa_2, Qb_2)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Qb_2 + Pb)

        # cascade 3
        Qa_3, Qb_3 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_3, Qb_3)

        return Za, Zb

    def forward_sa(self, Qa, Qb):
        Aa = self.proj1(Qa)  # 1*1卷积
        Ab = self.proj2(Qb)

        n, c, h, w = Aa.shape
        Aa = Aa.view(-1, h * w)
        Ab = Ab.view(-1, h * w)

        Aa = F.softmax(Aa, dim=1)
        Ab = F.softmax(Ab, dim=1)

        Aa = Aa.view(n, c, h, w)
        Ab = Ab.view(n, c, h, w)

        Qa_attened = Aa * Qa
        Qb_attened = Ab * Qb

        return Qa_attened, Qb_attened  # 此处还未交互

    def forward_co(self, Qa, Qb):
        Qa_low = self.proja(Qa)
        Qb_low = self.projb(Qb)

        N, C, H, W = Qa_low.shape
        Qa_low = Qa_low.view(N, C, H * W)
        Qb_low = Qb_low.view(N, C, H * W)
        Qb_low = torch.transpose(Qb_low, 1, 2)  # 矩阵转置相乘用于得到S矩阵

        L = torch.bmm(Qb_low, Qa_low)

        Aa = torch.tanh(L)
        Ab = torch.transpose(Aa, 1, 2)  # S矩阵转置用作权重

        N, C, H, W = Qa.shape

        Qa_ = Qa.view(N, C, H * W)
        Qb_ = Qb.view(N, C, H * W)

        Za = torch.bmm(Qb_, Aa)
        Zb = torch.bmm(Qa_, Ab)
        Za = Za.view(N, C, H, W)
        Zb = Zb.view(N, C, H, W)

        Za = F.normalize(Za)
        Zb = F.normalize(Zb)

        return Za, Zb


class MSGNet(nn.Module):
    def __init__(self):
        super(MSGNet, self).__init__()
        self.num_medium = 32
        ################################resnet101 Flow#######################################
        feats_Flow = models.resnet101(pretrained=True)
        self.conv0_Flow = nn.Sequential(feats_Flow.conv1, feats_Flow.bn1, nn.PReLU())
        self.conv1_Flow = nn.Sequential(feats_Flow.maxpool, *feats_Flow.layer1)
        self.conv2_Flow = feats_Flow.layer2
        self.conv3_Flow = feats_Flow.layer3
        self.conv4_Flow = feats_Flow.layer4

        ################################resnet101 RGB#######################################
        feats_RGB = models.resnet101(pretrained=True)
        self.conv0_RGB = nn.Sequential(feats_RGB.conv1, feats_RGB.bn1, nn.PReLU())
        self.conv1_RGB = nn.Sequential(feats_RGB.maxpool, *feats_RGB.layer1)
        self.conv2_RGB = feats_RGB.layer2
        self.conv3_RGB = feats_RGB.layer3
        self.conv4_RGB = feats_RGB.layer4

        self.atten_flow_channel_0 = ChannelAttention(64)
        self.atten_flow_spatial_0 = SpatialAttention()

        self.co_attention_0 = CoAttention(64)
        self.co_attention_1 = CoAttention(256)
        self.co_attention_2 = CoAttention(512)
        self.co_attention_3 = CoAttention(1024)
        self.co_attention_4 = CoAttention(2048)

        self.conv_01 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(32), nn.PReLU())
        self.conv_02 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(64), nn.PReLU())
        self.conv_11 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(32), nn.PReLU())
        self.conv_12 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(256), nn.PReLU())
        self.conv_21 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(32), nn.PReLU())
        self.conv_22 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(512), nn.PReLU())
        self.conv_31 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(32), nn.PReLU())
        self.conv_32 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1024, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(1024), nn.PReLU())
        self.conv_41 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(32), nn.PReLU())
        self.conv_42 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=2048, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(2048), nn.PReLU())

        self.attention_feature11 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                                 nn.PReLU())
        self.attention_feature21 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1),
                                                 nn.BatchNorm2d(512), nn.PReLU())
        self.attention_feature31 = nn.Sequential(nn.Conv2d(1024 * 2, 1024, kernel_size=3, padding=1),
                                                 nn.BatchNorm2d(1024), nn.PReLU())
        self.attention_feature41 = nn.Sequential(nn.Conv2d(2048 * 2, 2048, kernel_size=3, padding=1),
                                                 nn.BatchNorm2d(2048), nn.PReLU())

        self.attention_feature0 = nn.Sequential(nn.Conv2d(64 * 2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                                nn.PReLU(),
                                                nn.Conv2d(32, 2, kernel_size=3, padding=1))


        self.gate_RGB4 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.gate_RGB3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.gate_RGB2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.gate_RGB1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.gate_RGB0 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())

        self.gate_Flow4 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.gate_Flow3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.gate_Flow2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.gate_Flow1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.gate_Flow0 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())

        self.fuse4_Flow = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                        nn.PReLU(),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_RGB = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                       nn.PReLU(),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.channel4 = ChannelAttention(512)
        self.spatial4 = SpatialAttention()

        self.fuse3_Flow = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                        nn.PReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_RGB = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                       nn.PReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.channel3 = ChannelAttention(256)
        self.spatial3 = SpatialAttention()

        self.fuse2_Flow = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                        nn.PReLU(),
                                        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_RGB = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                       nn.PReLU(),
                                       nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.channel2 = ChannelAttention(128)
        self.spatial2 = SpatialAttention()

        self.fuse1_Flow = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_RGB = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.channel1 = ChannelAttention(64)
        self.spatial1 = SpatialAttention()

        self.fuse0_Flow = nn.Sequential(nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
                                        nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.fuse0_RGB = nn.Sequential(nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.channel0 = ChannelAttention(32)
        self.spatial0 = SpatialAttention()
        self.sigmoid2 = nn.Sigmoid()

        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
                                     nn.Conv2d(32, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, input, flow):
        c0_Flow = self.conv0_Flow(flow)  # N,64,192,192
        c1_Flow = self.conv1_Flow(c0_Flow)  # N,256,96,96
        c2_Flow = self.conv2_Flow(c1_Flow)  # N,512,48,48
        c3_Flow = self.conv3_Flow(c2_Flow)  # N,1024,24,24
        c4_Flow = self.conv4_Flow(c3_Flow)  # N,2048,12,12

        c0_RGB = self.conv0_RGB(input)  # N,64,192,192
        G0 = self.attention_feature0(torch.cat((c0_RGB, c0_Flow), dim=1))
        G0 = F.adaptive_avg_pool2d(torch.sigmoid(G0), 1)
        c0_RGB = G0[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * c0_RGB
        c0_Flow = G0[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * c0_Flow
        temp = c0_Flow.mul(self.atten_flow_channel_0(c0_Flow))
        temp = temp.mul(self.atten_flow_spatial_0(temp))
        c0_RGB = c0_RGB + temp

        c1_RGB = self.conv1_RGB(c0_RGB)  # N,256,96,96

        c2_RGB = self.conv2_RGB(c1_RGB)  # N,256,96,96
        co_att2_RGB, co_att2_Flow = self.co_attention_2(c2_RGB, c2_Flow)
        f21, f22 = self.co_attention_2(c2_RGB, c2_RGB)
        F2 = self.attention_feature21(torch.cat((co_att2_RGB, f21), dim=1))
        c2_RGB = F2 + c2_RGB

        c3_RGB = self.conv3_RGB(c2_RGB)  # N,256,96,96
        co_att3_RGB, co_att3_Flow = self.co_attention_3(c3_RGB, c3_Flow)
        f31, f32 = self.co_attention_3(c3_RGB, c3_RGB)
        F3 = self.attention_feature31(torch.cat((co_att3_RGB, f31), dim=1))
        c3_RGB = F3 + c3_RGB

        c4_RGB = self.conv4_RGB(c3_RGB)  # N,256,96,96
        co_att4_RGB, co_att4_Flow = self.co_attention_4(c4_RGB, c4_Flow)
        f41, f42 = self.co_attention_4(c4_RGB, c4_RGB)
        F4 = self.attention_feature41(torch.cat((co_att4_RGB, f41), dim=1))
        c4_RGB = F4 + c4_RGB
        ################################PAFEM######################################
        c4_RGB_512 = self.gate_RGB4(c4_RGB)  # 512
        c3_RGB_512 = self.gate_RGB3(c3_RGB)  # 256
        c2_RGB_512 = self.gate_RGB2(c2_RGB)  # 128
        c1_RGB_512 = self.gate_RGB1(c1_RGB)  # 64
        c0_RGB_512 = self.gate_RGB0(c0_RGB)  # 32

        c4_Flow_512 = self.gate_Flow4(c4_Flow)  # 512
        c3_Flow_512 = self.gate_Flow3(c3_Flow)  # 256
        c2_Flow_512 = self.gate_Flow2(c2_Flow)  # 128
        c1_Flow_512 = self.gate_Flow1(c1_Flow)  # 64
        c0_Flow_512 = self.gate_Flow0(c0_Flow)  # 32

        batch, channel, h, w = c4_RGB_512.shape
        M = h * w
        Flow_features4 = c4_Flow_512.view(batch, channel, M).permute(0, 2, 1)
        RGB_features4 = c4_RGB_512.view(batch, channel, M)
        p_4 = torch.matmul(Flow_features4, RGB_features4)
        p_4 = F.softmax((channel ** -.5) * p_4, dim=-1)
        feats_RGB4 = torch.matmul(p_4, RGB_features4.permute(0, 2, 1)).permute(0, 2, 1).view(batch, channel, h,
                                                                                             w)  # N,512,12,12

        E4_RGB = self.fuse4_RGB(torch.cat((c4_RGB_512, feats_RGB4), dim=1))  # 512->256 256->1 N,512,12,12
        E4_Flow = self.fuse4_Flow(torch.cat((c4_Flow_512, feats_RGB4), dim=1))  # 256->128
        temp_add = E4_RGB + E4_Flow
        channel_4 = self.channel4(temp_add)
        c4_attention = self.spatial4(channel_4 * temp_add)  # 4,1,12,12
        w_4 = self.sigmoid2(c4_attention)
        f4 = 2 * E4_RGB * w_4 + 2 * E4_Flow * (1 - w_4)
        feature_4 = f4 + feats_RGB4
        output1 = self.output1(feature_4)  # 512->256 4,256,12,12

        c3 = F.interpolate(output1, size=c3_RGB_512.size()[2:], mode='bilinear', align_corners=True)  # 256  4,256,24,24
        E3_RGB = self.fuse3_RGB(torch.cat((c3_RGB_512, c3), dim=1))  # 256->128
        E3_Flow = self.fuse3_Flow(torch.cat((c3_Flow_512, c3), dim=1))  # 256->128
        temp_add = E3_RGB + E3_Flow
        channel_3 = self.channel3(temp_add)
        c3_attention = self.spatial3(channel_3 * temp_add)  # 4,1,24,24
        w_3 = self.sigmoid2(c3_attention)
        f3 = 2 * E3_RGB * w_3 + 2 * E3_Flow * (1 - w_3)
        feature_3 = f3 + c3
        output2 = self.output2(feature_3)  # 512->256 4,256,12,12

        c2 = F.interpolate(output2, size=c2_RGB_512.size()[2:], mode='bilinear', align_corners=True)  # 256  4,128,48,48
        E2_RGB = self.fuse2_RGB(torch.cat((c2_RGB_512, c2), dim=1))  # 256->128
        E2_Flow = self.fuse2_Flow(torch.cat((c2_Flow_512, c2), dim=1))  # 256->128
        temp_add = E2_RGB + E2_Flow
        channel_2 = self.channel2(temp_add)
        c2_attention = self.spatial2(channel_2 * temp_add)  # 4,1,24,24
        w_2 = self.sigmoid2(c2_attention)
        f2 = 2 * E2_RGB * w_2 + 2 * E2_Flow * (1 - w_2)
        feature_2 = f2 + c2
        output3 = self.output3(feature_2)  # 256->128 4,64,48,48

        c1 = F.interpolate(output3, size=c1_RGB_512.size()[2:], mode='bilinear', align_corners=True)  # 256 4,64,96,96
        E1_RGB = self.fuse1_RGB(torch.cat((c1_RGB_512, c1), dim=1))  # 256->128
        E1_Flow = self.fuse1_Flow(torch.cat((c1_Flow_512, c1), dim=1))  # 256->128
        temp_add = E1_RGB + E1_Flow
        channel_1 = self.channel1(temp_add)
        c1_attention = self.spatial1(channel_1 * temp_add)  # 4,1,24,24
        w_1 = self.sigmoid2(c1_attention)
        f1 = 2 * E1_RGB * w_1 + 2 * E1_Flow * (1 - w_1)
        feature_1 = f1 + c1
        output4 = self.output4(feature_1)  # 256->128 4,32,96,96

        c0 = F.interpolate(output4, size=c0_RGB_512.size()[2:], mode='bilinear',
                           align_corners=True)  # 256  4,32,192,192
        E0_RGB = self.fuse0_RGB(torch.cat((c0_RGB_512, c0), dim=1))  # 256->128  4,32,192,192
        E0_Flow = self.fuse0_Flow(torch.cat((c0_Flow_512, c0), dim=1))  # 256->128
        temp_add = E0_RGB + E0_Flow
        channel_0 = self.channel0(temp_add)
        c0_attention = self.spatial0(channel_0 * temp_add)  # 4,1,192,192
        w_0 = self.sigmoid2(c0_attention)
        f0 = 2 * E0_RGB * w_0 + 2 * E0_Flow * (1 - w_0)
        feature_0 = f0 + c0
        output = self.output5(feature_0)  # 256->128 4,1,192,192

        output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)  # 4,1,384,384
        output = torch.sigmoid(output)

        return output, c4_attention, c3_attention, c2_attention, c1_attention, c0_attention


if __name__ == "__main__":
    model = MSGNet()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    depth = torch.autograd.Variable(torch.randn(4, 1, 384, 384))
    flow = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    output, a, b, c, d, e = model(input, flow)
    print(output.shape)
