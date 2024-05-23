import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import einops
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ------------------------------
# DATransformer Basic
# ------------------------------

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

# ------------------------------
# Swin Deformable Attention Transformer Basic
# ------------------------------

class DATSwinDAttention(nn.Module):
    r""" Shift Windows Deformable Attention

    Args:
        q_size(tuple[int]): Size of query. Here is the window size.
        kv_size(tuple[int]): Size of key and value. Here is the window size.
        dim (int): Number of input channels.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        window_size (tuple[int]): Window size for self-attention.
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        stride (int): Stride in offset calculation network
        offset_range_factor (int): Offset range factor in offset calculation network
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, q_size, kv_size, dim, n_head, n_group, window_size,
                 attn_drop, proj_drop, stride, offset_range_factor,
                 use_pe, dwc_pe, no_off, fixed_pe):
        super().__init__()

        self.dim = dim  # input channel
        self.window_size = window_size  # window height Wh, Window width Ww
        self.n_head = n_head  # number of head
        self.n_head_channels = self.dim // self.n_head  # head_dim
        self.scale = self.n_head_channels ** -0.5

        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size

        self.n_group = n_group
        self.n_group_channels = self.dim // self.n_group
        self.n_group_heads = self.n_head // self.n_group

        self.dwc_pe = dwc_pe
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        if self.q_h <= 12 or self.q_w <= 12:
            self.kk = 3
        elif 13 <= self.q_h <= 24 or 13 <= self.q_w <= 24:
            self.kk = 5
        elif 25 <= self.q_h <= 48 or 25 <= self.q_w <= 48:
            self.kk = 7
        else:
            self.kk = 9

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, stride, self.kk // 2,
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_group, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x, window_size, mask=None):
        H = window_size
        W = window_size
        B, N, C = x.size()
        dtype, device = x.dtype, x.device
        assert H * W == N, "input feature has wrong size"

        x = einops.rearrange(x, 'b (h w) c-> b c h w', h=H, w=W)
        # calculate query
        q = self.proj_q(x)  # B C H W
        # resize query
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_group, c=self.n_group_channels)

        # use query to calculate offset
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        # get the size of offset
        Hk, Wk = offset.size(2), offset.size(3)

        # sample number
        n_sample = Hk * Wk

        if self.offset_range_factor > 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk-1), 1.0 / (Wk-1)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_tmp = einops.rearrange(offset, '(b g) c h w -> b g c h w', g=self.n_group, c=2,h=4,w=4)
        # resize offset
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # use the number of offset point and batch size to get reference point
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        # no offset
        if self.no_off:
            offset = torch.zeros_like(offset)

        # offset + ref
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1,+1)#tanh()
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import cv2  
        # import imgviz
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import matplotlib.image as mpimg

        # pos_tmp = einops.rearrange(pos, '(b g) h w c -> b g c h w', g=self.n_group, c=2,h=4,w=4)

        # # 定义一个函数用于绘制形变场的轮廓图
        # def grid2contour(ax, grid):
        #     assert grid.ndim == 3
        #     x = np.arange(-1, 1, 2/grid.shape[1])
        #     y = np.arange(-1, 1, 2/grid.shape[0])
        #     X, Y = np.meshgrid(x, y)
        #     Z1 = grid[:,:,0] + 2  # 移除虚线
        #     Z1 = Z1[::-1]  # 垂直翻转
        #     Z2 = grid[:,:,1] + 2
            
        #     ax.contour(X, Y, Z1, 15, colors='k')
        #     ax.contour(X, Y, Z2, 15, colors='k')
        # original_image = mpimg.imread('/root/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/CIKM2017_dataset/test/sample_1139/img_1.png')
        # original_image=cv2.resize(original_image,(32,32))
        # # 创建一个大图
        # fig, ax = plt.subplots(int(np.sqrt(B)), int(np.sqrt(B)), figsize=(24,24), sharex=True, sharey=True)

        # # 调整子图之间的间隔为0
        # plt.subplots_adjust(wspace=0, hspace=0)

        # # 循环绘制每个小图
        # for i, flow in enumerate(pos_tmp[:,0,...].detach().cpu()):
        #     # 将光流场从Tensor格式转换为numpy格式
        #     flow = np.transpose(flow.numpy(),(1,2,0))
            
        #     # 计算当前小图的行和列索引
        #     row = i // int(np.sqrt(B))
        #     col = i % int(np.sqrt(B))

        #     # 使用箭头可视化光流
        #     grid2contour(ax[row, col], flow)

        #     # 计算当前切片的起始和结束行、列索引
        #     start_row = (i // (original_image.shape[1] // 4)) * 4
        #     end_row = start_row + 4
        #     start_col = (i % (original_image.shape[1] // 4)) * 4
        #     end_col = start_col + 4
            
        #     # 获取当前切片的图像数据
        #     slice_image = original_image[start_row:end_row, start_col:end_col]
        #     # ax[row, col].imshow(slice_image,alpha=0.5)  # 替换 original_image 为您的原始图像
        #     ax.imshow()
        #     # 关闭坐标轴
        #     ax[row, col].axis('off')

        # # 保存合成后的图像
        # plt.savefig('/root/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/zzwei_paper_code/models/datswinlstm_memory/pos.png')

        # # 创建一个大图
        # fig = plt.figure(figsize=(24,24))

        # # 使用gridspec来创建子图
        # gs = fig.add_gridspec(int(np.sqrt(B)), int(np.sqrt(B)),hspace=0,wspace=0)
        # pos_tmp = einops.rearrange(pos, '(b g) h w c -> b g c h w', g=self.n_group, c=2,h=4,w=4)
        # # 循环绘制每个小图
        # for i, flow in enumerate(pos_tmp[:,1,...].detach().cpu()):
        #     # 将光流场从Tensor格式转换为numpy格式
        #     flow = flow.numpy()
        #     magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
        #     angle = np.arctan2(flow[1], flow[0])

        #     # 计算当前小图的行和列索引
        #     row = i // int(np.sqrt(B))
        #     col = i % int(np.sqrt(B))

        #     # 在对应的位置创建子图
        #     ax = fig.add_subplot(gs[row, col])

        #     # 使用箭头可视化光流
        #     # grid2contour(np.transpose(flow,(1,2,0)))
        #     # ax.scatter(flow[0,...],flow[1,...])
        #     tmp=flow[0,...]+4
        #     ax.contour(flow[0,...], flow[1,...],tmp[::-1], colors='k')
        #     # # # 连接横向的点
        #     ax.contour(flow[0,...], flow[1,...],flow[1,...]+4, colors='k')

        #     # # 连接横向的点
        #     # for j in range(flow.shape[1]-1):
        #     #     ax.plot([flow[0,j], flow[0,j+1]], [flow[1,j], flow[1,j]], '-')

        #     # # 连接纵向的点
        #     # for j in range(flow.shape[0]-1):
        #     #     ax.plot([flow[0,j], flow[0,j]], [flow[1,j], flow[1,j+1]], '-')

        #     # 设置标题

        #     # 关闭坐标轴
        #     ax.axis('off')

        # # 调整布局
        # plt.tight_layout()

        # # 保存合成后的图像
        # plt.savefig('/root/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/zzwei_paper_code/models/datswinlstm_memory/pos.png')

# # 生成示例数据（576个4x4x2的矩阵）
# tensor = pos.cpu().numpy()

# # 创建一个空白的大矩阵用于存放所有子图
# big_matrix = np.zeros((24*4, 24*4))

# # 绘制大矩阵的子图
# fig, axes = plt.subplots(24, 24, figsize=(48, 48))

# # 遍历每个小矩阵，填充到对应的子图中，并画出格点位置
# for i in range(24):
#     for j in range(24):
#         index = i * 24 + j
#         # 将小矩阵的偏移量填充到大矩阵中相应的位置
#         big_matrix[i*4:(i+1)*4, j*4:(j+1)*4] = tensor[index, :, :, 0] + tensor[index, :, :, 1]
        
#         # 绘制子图，并画出格点位置
#         ax = axes[i, j]
#         #ax.imshow(tensor[index, :, :, 0], cmap='hot', interpolation='nearest')  # 画出x方向的偏移量
#         #x = np.arange(0.5, 4.5)  # x方向格点位置
#         #y = np.arange(0.5, 4.5)  # y方向格点位置
#         #X, Y = np.meshgrid(x, y)
#         ax.scatter(tensor[index, :, :, 0], tensor[index, :, :, 1], color='black', marker='o')  # 画出格点位置
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.axis('off')

# # 调整子图之间的间距
# plt.tight_layout()
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_group, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # embedding query,key,valuse B,C,H,W --> B*head,head_channel,HW
        q = q.reshape(B * self.n_head, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)

        # Q&K
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # use position encoding
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_head, self.n_head_channels,
                                                                              H * W)
            # fix
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_head, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(
                            B * self.n_group, H * W, 2).unsqueeze(2)
                        - pos.reshape(B * self.n_group, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_group, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_head, H * W, n_sample)
                attn = attn + attn_bias

        if mask is not None:
            attn = attn.view(-1, self.n_head, H * W, n_sample)
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.n_head, H * W, n_sample) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_head, H * W, n_sample)
            attn = attn.view(-1, H * W, n_sample)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y,pos.reshape(B, self.n_group, Hk, Wk, 2), reference.reshape(B, self.n_group, Hk, Wk, 2)#, 0, 0, 0

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, n_head={self.n_head}, n_group={self.n_group}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # self.proj_q(x)
        flops += N * self.dim * self.dim * 1
        if not self.no_off:
            # self.conv_offset(q_off)
            flops += self.n_group * N * self.n_group_channels * (
                    self.n_group_channels / self.n_group_channels) * self.kk * self.kk
            flops += self.n_group * N * self.n_group_channels
            flops += self.n_group * N * self.n_group_channels * 2 * 1
        # self.proj_k(x_sampled)
        flops += N * self.dim * self.dim * 1
        # self.proj_v(x_sampled)
        flops += N * self.dim * self.dim * 1
        # torch.einsum('b c m, b c n -> b m n', q, k)
        flops += self.n_group * N * self.n_group_channels * N
        # torch.einsum('b m n, b c n -> b c m', attn, v)
        flops += self.n_group * N * N * self.n_group_channels
        # self.proj_drop(self.proj_out(out))
        flops += N * self.dim * self.dim * 1
        return flops


class DATSwinTransformerBlock(nn.Module):
    r""" Swin Deformable Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LaerNorm
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, n_head, n_group, window_size, shift_size, mlp_ratio=2.,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.n_head = n_head
        self.n_head_channels = dim // n_head
        self.n_group = n_group
        self.window_size = window_size
        self.q_h, self.q_w = to_2tuple(window_size)
        self.kv_h, self.kv_w = to_2tuple(window_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = DATSwinDAttention(
            q_size=to_2tuple(window_size), kv_size=to_2tuple(window_size),
            dim=dim, n_head=n_head, n_group=n_group, window_size=to_2tuple(window_size),
            attn_drop=attn_drop, proj_drop=drop, stride=1, offset_range_factor=2,
            use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.red = nn.Linear(2*dim,dim)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x,hx=None):
        # H, W = x.size[]  # x (batch_in_each_GPU, H*W, embedding_channel)
        B, L, C = x.shape
        H,W = int(np.sqrt(L)),int(np.sqrt(L))
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x (batch_in_each_GPU, H*W, embedding_channel)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x,hx),-1)
            x = self.red(x)
        x = x.view(B, H, W, C)  # x (batch_in_each_GPU, embedding_channel, H, W)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x  # shifted_x (batch_in_each_GPU, embedding_channel, H, W)

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # (nW*B, window_size*window_size, C)  nW:number of Windows

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == (H,W):
            attn_windows, pos, ref = self.attn(x_windows, self.window_size,
                                                          mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows, pos, ref = self.attn(x_windows, self.window_size,
                                                          mask=self.calculate_mask((H,W)).to(
                                                              x.device))  # (nW*B, window_size*window_size, C)  nW:number of Windows

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         C)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C shifted_x  (batch_in_each_GPU, embedding_channel, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # Comment out this to calculation the time cost.
            # position = torch.roll(position, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            # reference = torch.roll(reference, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            # TODO ATTN_MAP SHIFT
        else:
            x = shifted_x

        x = x.view(B, H * W, C)  # x (batch_in_each_GPU, H*W, embedding_channel)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x#x, 0, 0, 0  # x (batch_in_each_GPU, H*W, embedding_channel)


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, n_head={self.n_head}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size  # nW: number of windows
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class DATSwinLayer(nn.Module):
    r""" Swin Deformable Attention Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        n_head (int): Number of attention heads.
        window_size (int): Local window size.
        n_group (int): Number of groups for offset.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Default: False.
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, depth, n_head, window_size, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.n_group = n_group
        self.use_checkpoint = use_checkpoint

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        # build blocks
        self.blocks = nn.ModuleList([
            DATSwinTransformerBlock(dim=dim,
                                   input_resolution=input_resolution,
                                   n_head=n_head,
                                   n_group=n_group,
                                   window_size=window_size,
                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   mlp_ratio=mlp_ratio,
                                   drop=drop,
                                   attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer,
                                   use_pe=use_pe,
                                   dwc_pe=dwc_pe,
                                   no_off=no_off,
                                   fixed_pe=fixed_pe,
                                   )
            for i in range(depth)])

        # patch merging/expend layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        # position = []
        # reference = []
        # attn_map = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x, _, __, ___ = checkpoint.checkpoint(blk, x, x_size)
                # position.append(pos)
                # reference.append(ref)
                # attn_map.append(att)
            else:
                x, _, __, ___ = blk(x, x_size)
                # position.append(pos)
                # reference.append(ref)
                # attn_map.append(att)

        if self.downsample is not None:
            x = self.downsample(x)
        # return x, position, reference, attn_map
        return x#, 0, 0, 0

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

