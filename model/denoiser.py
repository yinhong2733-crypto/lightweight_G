import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary


class Light_Residual_block (nn.Module):
    '''
    采用深度可分离卷积（DWConv）+ 点卷积（PWConv）的组合，减少参数量
    包含两个卷积序列，每个序列由深度卷积和点卷积组成
    加入残差连接（short_cut），当输入输出通道数或步长不同时使用 1x1 卷积调整维度
    '''
    def __init__(self, input_channels:int, output_channels:int,kernel_size:int,stride:int,dilation:int=1):
        super(Light_Residual_block, self).__init__()
        self.DWConv_1=nn.Conv2d(in_channels=input_channels, groups=input_channels,out_channels=input_channels,
                              kernel_size=kernel_size,stride=stride,padding=dilation*(kernel_size//2),padding_mode='reflect',
                                dilation=dilation,bias=False)
        self.PWConv_1=nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1,stride=1,padding=0,
                                bias=False)
        self.bn=nn.BatchNorm2d(output_channels)
        self.swish=nn.SiLU()
        self.DWConv_2=nn.Conv2d(in_channels=output_channels, groups=output_channels,out_channels=output_channels,
                                kernel_size=kernel_size,stride=1,padding=dilation*(kernel_size//2),padding_mode='reflect',
                                dilation=dilation,bias=False)
        self.PWConv_2=nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1,stride=1,padding=0,
                                bias=False)
        if (input_channels != output_channels) or (stride != 1):
            self.short_cut = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                       kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.short_cut = nn.Identity()

    def forward(self,x):
        out = self.DWConv_1(x)
        out = self.PWConv_1(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.DWConv_2(out)
        out = self.PWConv_2(out)
        out = out + self.short_cut(x)
        out = self.swish(out)

        return out


class Vascular_structure_enhancer(nn.Module):
    '''
    采用多尺度深度卷积（5x5、7x7、9x9、11x11）捕捉不同尺寸的血管特征
    将多尺度特征拼接后通过 1x1 卷积融合，增强血管结构信息
    '''
    def __init__(self, input_channels: int, output_channels: int):
        super(Vascular_structure_enhancer, self).__init__()
        self.CONv1_pre = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.DWCONv7_pre = nn.Conv2d(in_channels=output_channels, out_channels=output_channels,
                                     groups=output_channels,
                                     kernel_size=7, stride=1, padding=(7 // 2), padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.swish = nn.SiLU()
        self.DWCONv5 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, groups=output_channels,
                                 kernel_size=5, stride=1, padding=(5 // 2), padding_mode='reflect', bias=False)
        self.DWCONv7 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, groups=output_channels,
                                 kernel_size=7, stride=1, padding=(7 // 2), padding_mode='reflect', bias=False)
        self.DWCONv9 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, groups=output_channels,
                                 kernel_size=9, stride=1, padding=(9 // 2), padding_mode='reflect', bias=False)
        # self.DWCONv11 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, groups=output_channels,
        #                           kernel_size=11, stride=1, padding=(11 // 2), padding_mode='reflect', bias=False)
        self.identity = nn.Identity()
        self.CONv1_1 = nn.Conv2d(in_channels=(output_channels * 4), out_channels=output_channels, kernel_size=1,
                                 stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.CONv1_2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=1,
                                 stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.CONv1_pre(x)
        x = self.DWCONv7_pre(x)
        x = self.bn1(x)
        x = self.swish(x)
        x1 = self.DWCONv5(x)
        x2 = self.DWCONv7(x)
        x3 = self.DWCONv9(x)
        # x4 = self.DWCONv11(x)
        x5 = self.identity(x)
        out = torch.cat((x1, x2, x3, x5), dim=1)
        r1 = self.CONv1_1(out)
        r1 = self.bn2(r1)
        r1 = self.swish(r1)
        r2 = self.CONv1_2(r1)
        r2 = self.bn3(r2)
        r2 = self.swish(r2)

        return r2

class Residual_block(nn.Module):
    '''
    由两个 3x3 卷积组成，包含批归一化和 SiLU 激活函数
    实现基本的残差连接机制
    '''
    def __init__ (self,input_channels:int,output_channels:int,stride:int):
        super().__init__()
        self.residual_b1=nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,
                                   stride=stride,padding=(3//2),padding_mode='reflect',bias=False)
        self.residual_b2=nn.BatchNorm2d(output_channels)
        self.residual_b3=nn.SiLU()
        self.residual_b4=nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,
                                   stride=stride,padding=(3//2),padding_mode='reflect',bias=False)
        if input_channels != output_channels:
            self.residual_b5=nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=1,
                                       stride=stride,padding=0,bias=False)
        else:
            self.residual_b5=nn.Identity()
        self.residual_b6=nn.SiLU()

    def forward(self, x):
        out = self.residual_b1(x)
        out = self.residual_b2(out)
        out = self.residual_b3(out)
        out = self.residual_b4(out)
        skip = self.residual_b5(x)
        output = self.residual_b6(out + skip)

        return output


class Transformer_unit(nn.Module):
    '''
    修正：根据论文描述，Tn 包含 n 个残差块。
    T1 (Tn1) 对应最精细尺度，使用 1 个残差块。
    T2 (Tn2) 对应中间尺度，使用 2 个残差块。
    T3 (Tn3) 对应最粗糙尺度，使用 3 个残差块。
    '''

    def __init__(self):
        super().__init__()
        # # T1: 1个残差块 (96通道)
        # self.residual_block_1 = Residual_block(96, 96, stride=1)
        #
        # # T2: 2个残差块 (128通道)
        # self.residual_block_2 = nn.Sequential(
        #     Residual_block(128, 128, stride=1),
        #     Residual_block(128, 128, stride=1)
        # )
        #
        # # T3: 3个残差块 (160通道)
        # self.residual_block_3 = nn.Sequential(
        #     Residual_block(160, 160, stride=1),
        #     Residual_block(160, 160, stride=1),
        #     Residual_block(160, 160, stride=1)
        # )

        self.conv1_1 = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(in_channels=160, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, decoder_1, decoder_2, decoder_3):
        # 处理 T1 分支
        # tn1 = self.residual_block_1(decoder_1)
        tn1 = self.conv1_1(decoder_1)

        # 处理 T2 分支
        # tn2 = self.residual_block_2(decoder_2)
        tn2 = self.conv1_2(decoder_2)

        # 处理 T3 分支
        # tn3 = self.residual_block_3(decoder_3)
        tn3 = self.conv1_3(decoder_3)

        return tn1, tn2, tn3


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Light_Residual_block_1 = Light_Residual_block(input_channels=1,output_channels=96,kernel_size=3,stride=1,dilation=1)
        self.Light_Residual_block_2 = Light_Residual_block(input_channels=96,output_channels=128,kernel_size=3,stride=2,dilation=1)
        self.Light_Residual_block_3 = Light_Residual_block(input_channels=128,output_channels=160,kernel_size=3,stride=2,dilation=1)

    def forward(self,x):
        out_1 = self.Light_Residual_block_1(x)
        out_2 = self.Light_Residual_block_2(out_1)
        out_3 = self.Light_Residual_block_3(out_2)

        return out_1,out_2,out_3

class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.bridge = Light_Residual_block(input_channels=160,output_channels=192,kernel_size=3,stride=2,dilation=1)
    def forward(self,out_3):
        out = self.bridge(out_3)

        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.upsampler=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.decoder_1 = Light_Residual_block(input_channels=352, output_channels=160, kernel_size=3, stride=1,
                                              dilation=1)
        self.decoder_2 = Light_Residual_block(input_channels=288, output_channels=128, kernel_size=3, stride=1,
                                              dilation=1)
        self.decoder_3 = Light_Residual_block(input_channels=224, output_channels=96, kernel_size=3, stride=1,
                                              dilation=1)

    def forward(self, out, out_1, out_2, out_3):
        up_1 = F.interpolate(out, size=out_3.shape[-2:], mode='bilinear', align_corners=False)
        cat1 = torch.cat((up_1, out_3), dim=1)
        decoder_1 = self.decoder_1(cat1)

        up_2 = F.interpolate(decoder_1, size=out_2.shape[-2:], mode='bilinear', align_corners=False)
        cat2 = torch.cat((up_2, out_2), dim=1)
        # 修正：删除了这里重复的一行 cat2=torch.cat((up_2,out_2),dim=1)
        decoder_2 = self.decoder_2(cat2)

        up_3 = F.interpolate(decoder_2, size=out_1.shape[-2:], mode='bilinear', align_corners=False)
        cat3 = torch.cat((up_3, out_1), dim=1)
        decoder_3 = self.decoder_3(cat3)

        return decoder_1, decoder_2, decoder_3

class Denoiser(nn.Module):
    '''
    整合编码器、桥接层、解码器、血管增强器和转换单元
    输出三个不同尺度的特征图，可能用于多尺度血管结构的去噪和增强
    '''
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.bridge=Bridge()
        self.decoder=Decoder()
        # self.vascular_structure_enhancer=Vascular_structure_enhancer(input_channels=96,output_channels=96)
        self.transformer_unit=Transformer_unit()
    def forward(self,x):
        out1,out2,out3=self.encoder(x)
        bridge=self.bridge(out3)
        decoder_1,decoder_2,decoder_3=self.decoder(bridge,out1,out2,out3)
        # decoder_3=self.vascular_structure_enhancer(decoder_3)
        transformed_decoder_1, transformed_decoder_2, transformed_decoder_3=self.transformer_unit(decoder_3,decoder_2,decoder_1)


        return transformed_decoder_1,transformed_decoder_2,transformed_decoder_3


if __name__ == "__main__":
    # 实例化模型
    net = Denoiser().cpu()

    # 模拟输入
    x = torch.randn(1, 1, 224, 224)

    # 前向传播测试
    t1, t2, t3 = net(x)
    print("---------------------------------------")
    print(f"Model Output Shapes: {t1.shape}, {t2.shape}, {t3.shape}")
    print("---------------------------------------")

    # 统计参数量
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total Params: {total_params / 1e6:.4f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.4f} M")









