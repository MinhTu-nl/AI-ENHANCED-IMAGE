#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16 # Keep if using perception loss

# --- Model Definition ---
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)
        # Removed maxpool and upsample as they weren't used in the forward pass provided
        # self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1,x6],1))) # Changed F.tanh to torch.tanh
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)

        # Applying the enhancement curves
        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x) # Intermediate result might be useful
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x) # Final enhanced image
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1) # Concatenated curve parameters

        # Ensure output is within [0, 1] range if needed, often done after loss calculation or before saving/displaying
        # enhance_image = torch.clamp(enhance_image, 0.0, 1.0)

        return enhance_image_1, enhance_image, r

# --- Loss Function Definitions ---

# Color Constancy Loss (L_color)
class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return torch.mean(k) # Ensure scalar output

# Spatial Consistency Loss (L_spa)
class L_spa(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(L_spa, self).__init__()
        self.device = device
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4) # Pool size might need adjustment based on image size

    def forward(self, org , enhance ):
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        # Original paper might have specific weightings or calculations here.
        # This implementation follows the provided code structure
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up + D_down)

        return torch.mean(E) # Ensure scalar output

# Exposure Control Loss (L_exp)
class L_exp(nn.Module):
    def __init__(self,patch_size, mean_val, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        # E is the well-exposedness level, typically between 0.4 and 0.7
        self.mean_val = torch.FloatTensor([mean_val]).to(device)

    def forward(self, x ):
        x = torch.mean(x,1,keepdim=True) # Average across color channels
        mean = self.pool(x) # Average pooling across patches
        d = torch.mean(torch.pow(mean - self.mean_val, 2)) # MSE loss from target mean
        return d

# Total Variation Loss (L_TV)
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight * (h_tv/count_h + w_tv/count_w) / batch_size # Removed factor of 2 compared to original code

# Illumination Smoothness Loss (Used on the 'r' maps) - This seems to be L_TV applied to 'r'
class L_TV_R(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV_R, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.l_tv = L_TV(TVLoss_weight=TVLoss_weight)

    def forward(self, r):
        # The 'r' tensor has shape (batch, 24, H, W). We need to calculate TV loss for each channel.
        # A common approach in Zero-DCE is to apply TV loss to each of the 8 predicted 3-channel 'r' maps.
        loss = 0
        for i in range(8): # Assuming 8 iterations as in the forward pass
            # Extract the i-th set of 3 channels (r_i map)
            r_map = r[:, i*3:(i+1)*3, :, :]
            loss += self.l_tv(r_map)
        return loss / 8 # Average the loss over the 8 maps


# --- Utility Functions (Optional, can be moved to train.py) ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu') # Kaiming init often works well
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Note: Sa_Loss and perception_loss were in the original file but not used in the Zero-DCE paper's loss formulation.
# They are omitted here for simplicity, focusing on the core Zero-DCE losses (L_spa, L_exp, L_col, L_TV applied to 'r').
# If you need them, they can be added back.