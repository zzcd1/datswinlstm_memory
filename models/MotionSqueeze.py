import torch
import torch.nn as nn
from spatial_correlation_sampler import SpatialCorrelationSampler
import torch.nn.functional as F


class MS(nn.Module):   # # Multi-scale Temporal Dynamics Module
## https://github.com/yzfly/TCM/blob/main/TCM.py
    def __init__(self, expansion = 1, pos=2,c_out=512):
        super(MS, self).__init__()
        patchs = [7, 7, 7, 7] # 25 25 13 7
        self.patch = patchs[pos-1]
        self.patch_dilation = 1
        self.soft_argmax = nn.Softmax(dim=1)
        self.expansion = expansion
        self.c_out = c_out
        
        self.matching_layer = Matching_layer(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)   
        # self.matching_layer = Matching_layer_mm(patch=self.patch)   
        
        self.c1 = 16
        self.c2 = 32
        self.c3 = 64
        self.flow_refine_conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3, bias=False),
            # nn.BatchNorm2d(3),
            nn.SiLU(),
            nn.Conv2d(3, self.c1, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(self.c1),
            nn.SiLU()
        )
        self.flow_refine_conv2 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False),
            # nn.BatchNorm2d(self.c1),
            nn.SiLU(),
            nn.Conv2d(self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(self.c2),
            nn.SiLU(),
        )
        self.flow_refine_conv3 = nn.Sequential(
            nn.Conv2d(self.c2, self.c2, kernel_size=3, stride=1, padding=1, groups=self.c2, bias=False),
            # nn.BatchNorm2d(self.c2),
            nn.SiLU(),
            nn.Conv2d(self.c2, self.c3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(self.c3),
            nn.SiLU(),
        )
        self.flow_refine_conv4 = nn.Sequential(
            nn.Conv2d(self.c3, self.c3, kernel_size=3, stride=1, padding=1, groups=self.c3, bias=False),
            # nn.BatchNorm2d(self.c3),
            nn.SiLU(),
            nn.Conv2d(self.c3, self.c_out, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(self.out_channel),
            nn.SiLU(),
        )



    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm) 
    
    def apply_binary_kernel(self, match, h, w, region):
        # binary kernel
        x_line = torch.arange(w, dtype=torch.float).to(match.device).detach()
        y_line = torch.arange(h, dtype=torch.float).to(match.device).detach()
        x_kernel_1 = x_line.view(1,1,1,1,w).expand(1,1,w,h,w).to(match.device).detach()
        y_kernel_1 = y_line.view(1,1,1,h,1).expand(1,h,1,h,w).to(match.device).detach()
        x_kernel_2 = x_line.view(1,1,w,1,1).expand(1,1,w,h,w).to(match.device).detach()
        y_kernel_2 = y_line.view(1,h,1,1,1).expand(1,h,1,h,w).to(match.device).detach()

        ones = torch.ones(1).to(match.device).detach()
        zeros = torch.zeros(1).to(match.device).detach()

        eps = 1e-6
        kx = torch.where(torch.abs(x_kernel_1 - x_kernel_2)<=region, ones, zeros).to(match.device).detach()
        ky = torch.where(torch.abs(y_kernel_1 - y_kernel_2)<=region, ones, zeros).to(match.device).detach()
        kernel = kx * ky + eps
        kernel = kernel.view(1,h*w,h*w).to(match.device).detach()                
        return match* kernel


    def apply_gaussian_kernel(self, corr, h,w,p, sigma=5):
        b, c, s = corr.size()

        x = torch.arange(p, dtype=torch.float).to(corr.device).detach()
        y = torch.arange(p, dtype=torch.float).to(corr.device).detach()

        idx = corr.max(dim=1)[1] # b x hw    get maximum value along channel
        idx_y = (idx // p).view(b, 1, 1, h, w).float()
        idx_x = (idx % p).view(b, 1, 1, h, w).float()
        # UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
        x = x.view(1,1,p,1,1).expand(1, 1, p, h, w).to(corr.device).detach()
        y = y.view(1,p,1,1,1).expand(1, p, 1, h, w).to(corr.device).detach()

        gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
        gauss_kernel = gauss_kernel.view(b, p*p, h*w)#.permute(0,2,1).contiguous()

        return gauss_kernel * corr

    def match_to_flow_soft(self, match, k, h,w, temperature=1, mode='softmax'):        
        b, c , s = match.size()     
        idx = torch.arange(h*w, dtype=torch.float32).to(match.device)
        idx_x = idx % w
        idx_x = idx_x.repeat(b,k,1).to(match.device)
        idx_y = torch.floor(idx / w)   
        idx_y = idx_y.repeat(b,k,1).to(match.device)

        soft_idx_x = idx_x[:,:1]
        soft_idx_y = idx_y[:,:1]
        displacement = (self.patch-1)/2
        
        topk_value, topk_idx = torch.topk(match, k, dim=1)    # (B*T-1, k, H*W)
        topk_value = topk_value.view(-1,k,h,w)
        
        match = self.apply_gaussian_kernel(match, h, w, self.patch, sigma=5)
        match = match*temperature
        match_pre = self.soft_argmax(match)
        smax = match_pre           
        smax = smax.view(b,self.patch,self.patch,h,w)  # 相当于个概率，最大是1，代表最有可能移动的方向
        x_kernel = torch.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=torch.float).to(match.device)
        y_kernel = torch.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=torch.float).to(match.device)
        x_mult = x_kernel.expand(b,self.patch).view(b,self.patch,1,1)
        y_mult = y_kernel.expand(b,self.patch).view(b,self.patch,1,1)
            
        smax_x = smax.sum(dim=1, keepdim=False) #(b,w=k,h,w)
        smax_y = smax.sum(dim=2, keepdim=False) #(b,h=k,h,w)
        flow_x = (smax_x*x_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)
        flow_y = (smax_y*y_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)    

        flow_x = (flow_x / (self.patch_dilation * displacement)) # 
        flow_y = (flow_y / (self.patch_dilation * displacement))
            
        return flow_x, flow_y, topk_value     

    def flow_computation(self, x, pos=0, temperature=100):
        
        size = x.size()               
        # x = x.view((-1, self.num_segments) + size[1:])        # N T C H W
        # x = x.permute(0,2,1,3,4).contiguous() # B C T H W   
                        
        # match to flow            
        k = 1         
        temperature = temperature       
        b,c,t,h,w = x.size()            
        t = t-1         
        # if pos == 0:
        #     x_pre = x[:,:,0,:].unsqueeze(dim=2).expand((b,c,t,h,w)).permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
        # else:
        #     x_pre = x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
            
        #x_pre = x[:,:,0,:].unsqueeze(dim=2).expand((b,c,t-1,h,w))
        x_pre = x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
        x_post = x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)

        match = self.matching_layer(x_pre, x_post)    # (B*T-1*group, H*W, H*W)          
        u, v, confidence = self.match_to_flow_soft(match, k, h, w, temperature) # u/v:b,1,h*w
        flow = torch.cat([u,v], dim=1).view(-1, 2*k, h, w)  #  (b, 2, h, w)  
    
        return flow, confidence

    def forward(self,x):
        b,c_in,t,h,w = x.shape
        flow_1, match_v1 = self.flow_computation(x, pos=1)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import cv2  
        # import imgviz
        # fig, axes = plt.subplots(1, 11, figsize=(15, 3))  # 创建一个子图，列数等于图像序列长度
        # for i, (flow,conf) in enumerate(zip(flow_1.detach().cpu(),match_v1.detach().cpu())):
        #     conf = conf.permute(1, 2, 0)
        #     flow[0,:,:][conf[:,:,0]<0.01]=0
        #     flow[1,:,:][conf[:,:,0]<0.01]=0

        #     # 将光流场从Tensor格式转换为numpy格式
        #     flow = flow.permute(1,2,0).numpy()
        #     flow = imgviz.flow2rgb(flow)
        #     # # 计算光流的大小和方向
        #     # magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
        #     # angle = np.arctan2(flow[1], flow[0])
        #     # # 使用箭头可视化光流
        #     axes[i].imshow(flow)  
        #     axes[i].axis('off')
        #     # axes[i].quiver(np.linspace(-np.max(magnitude), np.max(magnitude),24), np.linspace(-np.max(magnitude), np.max(magnitude),24), flow[0, ::2, ::2], flow[1, ::2, ::2], angles='xy', scale_units='xy', scale=0.5, color='r')
        #     axes[i].set_title(f" Learned Flow {i+1}")  # 设置标题
        # plt.savefig('/home/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/zzwei_paper_code/models/datswinlstm_memory/MotionSqueezeFlow1.png')
        # import cv2
        # fig, axes = plt.subplots(1, 11, figsize=(15, 3))  # 创建一个子图，列数等于图像序列长度
        # for i, img in enumerate(match_v1.detach().cpu()):
        #     # 将图像从Tensor格式转换为numpy格式，并交换通道顺序
        #     img = img.permute(1, 2, 0).numpy()
        #     # 显示图像
        #     axes[i].imshow(np.squeeze(img,2),cmap='hot',vmin=0,vmax=0.2)#,origin='lower'
        #     axes[i].axis('off')  # 关闭坐标轴
        #     axes[i].set_title(f"Confidence Map {i+1}")  # 设置标题
        # plt.savefig('/home/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/zzwei_paper_code/models/datswinlstm_memory/MotionSqueeze1.png')

        # # flow_2, match_v2 = self.flow_computation(x, pos=0)
        
        x1 = torch.cat([flow_1, match_v1], dim=1)
        # import numpy as np
        # np.save('/workspace/swinlstm/SwinLSTM-main/motion.npy',x1.detach().cpu.numpy())
        # print(100*'*')
        # # x2 = torch.cat([flow_2, match_v2], dim=1)

        _, c, h, w = x1.size()
        x1 = x1.view(-1,(t-1),c,h,w)
        # x2 = x2.view(-1,self.num_segments-1,c,h,w)

        x1 = torch.cat([x1,x1[:,-1:,:,:,:]], dim=1) ## (b,t,3,h,w) 保持原始时间序列长度t不变
        x1 = x1.view(-1,c,h,w)

        x1 = self.flow_refine_conv1(x1)
        x1 = self.flow_refine_conv2(x1)
        x1 = self.flow_refine_conv3(x1)
        x1 = self.flow_refine_conv4(x1)
        x1 = x1.view(b,c_in,t,h,w)
        x1 = x1 + x
        return x1

class Matching_layer(nn.Module):
    def __init__(self, ks, patch, stride, pad, patch_dilation):
        super(Matching_layer, self).__init__()
        self.act = nn.SiLU()
        self.patch = patch
        self.correlation_sampler = SpatialCorrelationSampler(ks, patch, stride, pad, patch_dilation)
        
    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        corr = self.correlation_sampler(feature1, feature2)
        corr = corr.view(b, self.patch * self.patch, h1* w1) # Channel : target // Spatial grid : source
        corr = self.act(corr)
        return corr



class Matching_layer_mm(nn.Module):
    def __init__(self, patch):
        super(Matching_layer_mm, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.patch = patch
    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)
    def corr_abs_to_rel(self,corr,h,w):
        max_d = self.patch // 2
        b,c,s = corr.size()        
        corr = corr.view(b,h,w,h,w)
        w_diag = torch.zeros((b,h,h,self.patch ,w),device=corr.device)
        for i in range(max_d+1):
            if (i==0):
                w_corr_offset = torch.diagonal(corr,offset=0,dim1=2,dim2=4)       
                w_diag[:,:,:,max_d] = w_corr_offset
            else:
                w_corr_offset_pos = torch.diagonal(corr,offset=i,dim1=2,dim2=4) 
                w_corr_offset_pos = F.pad(w_corr_offset_pos, (i,0)) #.unsqueeze(5)
                w_diag[:,:,:,max_d-i] = w_corr_offset_pos
                w_corr_offset_neg = torch.diagonal(corr,offset=-i,dim1=2,dim2=4) 
                w_corr_offset_neg = F.pad(w_corr_offset_neg, (0,i)) #.unsqueeze(5)
                w_diag[:,:,:,max_d+i] = w_corr_offset_neg
        hw_diag = torch.zeros((b,self.patch ,w,self.patch ,h),device=corr.device) 
        for i in range(max_d+1):
            if (i==0):
                h_corr_offset = torch.diagonal(w_diag,offset=0,dim1=1,dim2=2)
                hw_diag[:,:,:,max_d] = h_corr_offset
            else:
                h_corr_offset_pos = torch.diagonal(w_diag,offset=i,dim1=1,dim2=2) 
                h_corr_offset_pos = F.pad(h_corr_offset_pos, (i,0)) #.unsqueeze(5)
                hw_diag[:,:,:,max_d-i] = h_corr_offset_pos
                h_corr_offset_neg = torch.diagonal(w_diag,offset=-i,dim1=1,dim2=2) 
                h_corr_offset_neg = F.pad(h_corr_offset_neg, (0,i)) #.unsqueeze(5)      
                hw_diag[:,:,:,max_d+i] = h_corr_offset_neg                
        hw_diag = hw_diag.permute(0,3,1,4,2).contiguous()
        hw_diag = hw_diag.view(-1,self.patch *self.patch ,h*w)      
        return hw_diag    

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        feature1 = feature1.view(b, c, h1 * w1)
        feature2 = feature2.view(b, c, h2 * w2)
        corr = torch.bmm(feature2.transpose(1, 2), feature1)
        corr = corr.view(b, h2 * w2, h1 * w1) # Channel : target // Spatial grid : source
        corr = self.corr_abs_to_rel(corr,h1,w1) # (b,pp,hw)
        corr = self.relu(corr)
        return corr

if __name__ == "__main__":
    device = torch.device('cuda:3')
    x = torch.rand(1,256,12,12,12).to(device)
    motion_model = MS(expansion = 1, pos=2).to(device)   # # Multi-scale Temporal Dynamics Module
    y=motion_model(x)
    print(x.shape)


