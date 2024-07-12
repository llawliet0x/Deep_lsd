# import torch
# import torch.nn as nn
# import numpy as np


# class fire_module(nn.Module):
#     def __init__(self, c_in, c_out_p, c_out, s, transpose=False):
#         super(fire_module, self).__init__()

#         self.conv_1 = nn.Sequential( nn.Conv2d(c_in, c_out_p, kernel_size=1), nn.BatchNorm2d(c_out_p), nn.ReLU6(inplace=True) )
        
#         self.conv_2 = nn.Sequential( nn.Conv2d(c_out_p, int(c_out/2), kernel_size=3, stride=s, padding=1), nn.BatchNorm2d(int(c_out/2)), nn.ReLU6(inplace=True) )
#         self.conv_3 = nn.Sequential( nn.Conv2d(c_out_p, int(c_out/2), kernel_size=1, stride=s), nn.BatchNorm2d(int(c_out/2)), nn.ReLU6(inplace=True) )

#         self.conv_2_T = nn.Sequential( nn.ConvTranspose2d(c_out_p, int(c_out/2), kernel_size=2, stride=2), nn.BatchNorm2d(int(c_out/2)), nn.ReLU6(inplace=True) )
#         self.conv_3_T = nn.Sequential( nn.ConvTranspose2d(c_out_p, int(c_out/2), kernel_size=1, stride=2), nn.BatchNorm2d(int(c_out/2)), nn.ReLU6(inplace=True) )

#         self.trnaspose = transpose


#     def forward(self,x):
#         if self.trnaspose:
#             x = self.conv_1(x)
#             x_1 = self.conv_2_T(x)
#             x_2 = self.conv_3_T(x)
#             x_2 = nn.functional.pad(x_2, (0,1,0,1)) 
#             x = torch.cat([x_1, x_2], dim=1)
#         else:
#             x = self.conv_1(x)
#             x_1 = self.conv_2(x)
#             x_2 = self.conv_3(x)
#             x = torch.cat([x_1, x_2], dim=1)
        
#         return x
    



# class convolution_block(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(convolution_block, self).__init__()
#         self.conv_1 = nn.Sequential( nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU6(inplace=True) )
#         self.conv_2 = nn.Sequential( nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU6(inplace=True) )
#     def forward(self,x):
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         return x 
    



# class Down_Sample(nn.Module):
#     def __init__(self, c_in, c_out_p, c_out):
#         super(Down_Sample, self).__init__()
#         self.fire_module_1 =fire_module(c_in, c_out_p, c_out, 2)
#         self.fire_module_2 = fire_module(c_out, c_out_p, c_out, 1)
#     def forward(self,x):
#         x = self.fire_module_1(x)
#         x = self.fire_module_2(x)
#         return x 




# class Up_Sample(nn.Module):
#     def __init__(self, c_in, c_out_p1, c_out_p2, c_out):
#         super(Up_Sample, self).__init__()
#         self.fire_module_T = fire_module(c_in, c_out_p1, c_out, 2 , True)
#         self.fire_module_1 = fire_module(c_out*2, c_out_p2, c_out, 1)
#         self.fire_module_2 = fire_module(c_out, c_out_p2, c_out, 1)
#     def forward(self,x,y):
#         x = self.fire_module_T(x)
#         x = torch.cat([x,y], dim=1)
#         x = self.fire_module_1(x)
#         x = self.fire_module_2(x)
#         return x




# class SqueezeUNet(nn.Module):
#     def __init__(self, c_in = 1, c_out = 32):
#         super(SqueezeUNet, self).__init__()
#         self.convblock_1 = convolution_block(c_in, 64)
        
#         self.DS1 = Down_Sample(64, 32, 128)
#         self.DS2 = Down_Sample(128, 48, 256)
#         #self.DS3 = Down_Sample(256, 64, 512)
#         #self.DS4 = Down_Sample(512, 80, 1024)

#         #self.US1 = Up_Sample(1024, 80, 64, 512)
#         #self.US2 = Up_Sample(512, 64, 48, 256)
#         self.US3 = Up_Sample(256, 48, 32, 128)

#         self.convT = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU6(inplace=True) )
#         self.convblock_2 = convolution_block(128, 64)
#         self.conv_out = nn.Conv2d(64, c_out, 1)


#     def forward(self,x):
#         x0 = self.convblock_1(x)
#         x1 = self.DS1(x0)
#         x2 = self.DS2(x1)
#         #x3 = self.DS3(x2)
#        #x4 = self.DS4(x3)

#         # print(f'x0 = {x0.size()}')
#         # print(f'x1 = {x1.size()}')
#         # print(f'x2 = {x2.size()}')
#         # print(f'x3 = {x3.size()}')
#         # print(f'x4 = {x4.size()}')

#         #x = self.US1(x4, x3)
#         #print(x.size())
#         #x = self.US2(x3, x2)
#         #print(x.size())
#         x = self.US3(x2, x1)
#         #print(x.size())
#         x = self.convT(x)
#         #print(x.size())

#         x = torch.cat([x,x0], dim=1)
#         x = self.convblock_2(x)
#         x = self.conv_out(x)

#         return x 


import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.squeeze(x)
        x = F.relu(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)

class SqueezeUNet(nn.Module):
    def __init__(self, num_classes = 32):
        super(SqueezeUNet, self).__init__()
        self.encoder1 = FireModule(1, 16, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = FireModule(128, 32, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.decoder1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        
        self.final = nn.Conv2d(128, num_classes, kernel_size=1)
    
    # def forward(self, x):
    #     enc1 = self.encoder1(x)
    #     enc2 = self.encoder2(self.pool1(enc1))
        
    #     dec1 = self.decoder1(self.pool2(enc2))
    #     dec2 = self.decoder2(dec1 + enc2)
        
    #     return self.final(dec2 + enc1)
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        
        dec1 = self.decoder1(self.pool2(enc2))
        dec1_padded = self.pad_to_match(dec1, enc2)
        
        dec2 = self.decoder2(dec1_padded + enc2)
        dec2_padded = self.pad_to_match(dec2, enc1)
        
        return self.final(dec2_padded + enc1)    

    def pad_to_match(self, src, target):
        """
        Pad the src tensor to match the spatial dimensions of the target tensor.
        """
        src_shape = src.size()[2:]
        target_shape = target.size()[2:]
        padding = [
            (target_shape[1] - src_shape[1]) // 2, (target_shape[1] - src_shape[1] + 1) // 2,
            (target_shape[0] - src_shape[0]) // 2, (target_shape[0] - src_shape[0] + 1) // 2
        ]
        return F.pad(src, padding)