import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from MotionSqueeze import MS
import torch.utils.checkpoint as checkpoint
from dat_blocks import DATSwinLayer, DATSwinTransformerBlock
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table


### 基于3D卷积
class MotionEncoder3D(nn.Module):
    def __init__(self):
        super(MotionEncoder3D ,self).__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1 ,32 ,kernel_size=(3 ,7 ,7) ,stride=(1 ,2 ,2) ,padding=(1 ,3 ,3)),
            nn.SiLU())
        self.conv3d_2 = nn.Sequential(nn.Conv3d(32 ,64 ,kernel_size=(3 ,3 ,3) ,stride=(1 ,2 ,2) ,padding=(0 ,1 ,1)),
                                      nn.SiLU())
        self.conv3d_3 = nn.Sequential(nn.Conv3d(64 ,128 ,kernel_size=(3 ,3 ,3) ,stride=(1 ,2 ,2) ,padding=(1 ,1 ,1)),
                                      nn.SiLU())
        self.conv3d_4 = nn.Sequential(nn.Conv3d(128 ,512 ,kernel_size=(3 ,3 ,3) ,stride=(1 ,2 ,2) ,padding=(0 ,1 ,1)),
                                      nn.SiLU())
        # self.linear = nn.Sequential(nn.Linear(256,240),
        #                             nn.SiLU() )## 240, memory_size)
        self.avg_pool = nn.AdaptiveAvgPool3d([1 ,6 ,6])

    def forward(self ,x):
        x = x.permute(0 ,2 ,1 ,3 ,4)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        x = self.avg_pool(x)
        # x = self.linear(x)
        return x.squeeze(2)

    # class MotionEncoder2D(nn.Module):
    #    def __init__(self, in_len=12, out_c=400):
    #        super(MotionEncoder2D, self).__init__()
    #        self.conv2d_1 = nn.Sequential(
    #            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
    #            nn.SiLU())
    #        self.conv2d_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    #                                     nn.SiLU())
    #        self.conv2d_3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    #                                      nn.SiLU())
    #        self.conv2d_4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    #                                      nn.SiLU())
    #        self.conv2d_5 = nn.Sequential(nn.Conv2d(256, out_c, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    #                                      nn.SiLU())
    #        self.conv3d_1 = nn.Sequential(
    #            nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
    #            nn.SiLU())
    #        self.conv3d_2 = nn.Sequential(
    #            nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
    #            nn.SiLU())
    #        self.avg_pool_3d = nn.AdaptiveAvgPool3d([1, None, None])

    # def forward(self, x):
    #     # print(f"MotionEncoder2D Input: {x.shape}")
    #     b, t, c, h, w = x.shape
    #     x = x.contiguous().view(-1, c, h, w)  # b*t, c, h, w
    #     x = self.conv2d_1(x)
    #     # print(f"After conv2d_1: {x.shape}")
    #     x = self.conv2d_2(x)
    #     # print(f"After conv2d_2: {x.shape}")
    #     x = self.conv2d_3(x)
    #     # print(f"After conv2d_3: {x.shape}")
    #     x = self.conv2d_4(x)
    #     # print(f"After conv2d_4: {x.shape}")
    #     x = self.conv2d_5(x)
    #     # print(f"After conv2d_5: {x.shape}")
    #     x = x.contiguous().view(b, -1, t, h // 32, w // 32)
    #     # print(f"After view: {x.shape}")
    #     motion_features = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    #     zero_frame = torch.zeros_like(x[:, :, :1, :, :])
    #     motion_features = torch.cat([zero_frame, motion_features], dim=2)
    #     x = self.conv3d_1(motion_features)
    #     # print(f"After conv3d_1: {x.shape}")
    #     x = self.conv3d_2(x)
    #     # print(f"After conv3d_2: {x.shape}")
    #     x = self.avg_pool_3d(x)
    #     # print(f"After avg_pool_3d: {x.shape}")
    #     return x.squeeze(2)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=128, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim * 1, bias=False)
        self.to_k = nn.Linear(dim, inner_dim * 1, bias=False)
        self.to_v = nn.Linear(dim, inner_dim * 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y, z):
        if x.equal(y):
            x = self.norm(x)
            y = self.norm(y)
            z = self.norm(z)
        else:
            x = self.norm(x)
            y = self.norm(y)

        qkv = [self.to_q(x), self.to_k(y), self.to_v(z)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, y, z):
        for attn_1, attn_2, ff in self.layers:
            x = attn_1(x=x, y=x, z=x) + x
            x = attn_2(x=x, y=y, z=z) + x
            x = ff(x) + x
        return torch.sigmoid(self.norm(x))


class Memory(nn.Module):
    def __init__(self, args, memory_channel_size=256, memory_slot_size=100, short_len=24, long_len=37):
        super(Memory, self).__init__()
        self.args = args
        self.memory_size = memory_channel_size
        self.memory_slot_size = memory_slot_size
        self.short_len = short_len
        self.long_len = long_len
        # self.motion_encoder_2d = MotionEncoder2D(in_len=self.short_len, out_c=self.memory_size)
        # self.motion_context_encoder_2d = MotionEncoder2D(in_len=self.long_len, out_c=self.memory_size)
        ## 使用新的motion_encoder
        self.motion_encoder_2d = MotionEncoder3D()
        self.motion_context_encoder_2d = MotionEncoder3D()

        self.pos_embedding = nn.Parameter(torch.randn(1, 6 * 6, self.memory_size))
        self.additional_linear = nn.Linear(self.memory_size, 512)
        self.embedder = nn.Sequential(
            PatchExpanding(input_resolution=[6, 6], dim=512, dim_scale=2),
            PatchExpanding(input_resolution=[12, 12], dim=256, dim_scale=2),
            PatchExpanding(input_resolution=[24, 24], dim=128, dim_scale=2),
            nn.Linear(64, 256),
        )

        self.attention_block = Transformer(dim=self.memory_size, depth=1, heads=8, dim_head=64,
                                           mlp_dim=self.memory_size, dropout=0.1)
        self.predictor = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                                  in_chans=args.input_channels, embed_dim=args.embed_dim,
                                  depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                                  num_heads=args.heads_number, window_size=args.window_size)

        self.target_len = args.out_len
        self.memory_shape = [self.memory_slot_size, self.memory_size]
        self.memory_bank = nn.init.uniform_(torch.empty(self.memory_shape), a=0.0, b=1.0)
        self.memory_bank = nn.Parameter(self.memory_bank, requires_grad=True)
        self.attention_size = 256
        self.attention_func = nn.Sequential(
            nn.Linear(512, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.attention_size),
            nn.Sigmoid())
        self.fuse = nn.Linear(512, 256)

    def forward(self, inputs, memory_x, phase):
        # print(f"Memory Input: inputs {inputs.shape}, memory_x {memory_x.shape}, phase {phase}")
        ## 用帧差作为运动信息
        memory_x = memory_x[: ,1: ,...] - memory_x[: ,:-1 ,...] # B,T,C,H,W
        # print(memory_x.shape)
        b, t, c, h, w = memory_x.shape
        memory_query = self.motion_context_encoder_2d(memory_x) if phase == 1 else self.motion_encoder_2d(memory_x)
        # print(f"After motion_encoder: {memory_query.shape}")

        b_, c_, h_, w_ = memory_query.shape
        memory_query = memory_query.contiguous().permute(0, 2, 3, 1).view(b_, -1, c_)
        # print(f"After permute and view: {memory_query.shape}")

        b__, l, c__ = memory_query.shape
        memory_bank = self.memory_bank.unsqueeze(0).repeat(b_, 1, 1).contiguous()
        memory_query += self.pos_embedding
        matched_memory = self.attention_block(memory_query, memory_bank, memory_bank)
        # print(f"After attention_block: {matched_memory.shape}")

        matched_memory = self.embedder(matched_memory)
        # print(f"After embedder: {matched_memory.shape}")

        outputs = []
        inputs_len = inputs.shape[1]
        last_input = inputs[:, -1]

        states_down = [None] * len(self.args.depths_down)
        states_up = [None] * len(self.args.depths_down)

        for i in range(inputs_len - 1):
            output, states_down, states_up = self.predictor(inputs[:, i], states_down, states_up)
            # print(f"Predictor output at step {i}: {output.shape}")
            outputs.append(output)
        # print(f"states_down shape: {states_down[-1][1].shape}")
        # print(f"matched_memory shape: {matched_memory.shape}")
        for i in range(self.target_len):
            output, states_down, states_up = self.predictor(last_input, states_down, states_up)
            # print(f"Predictor output at target step {i}: {output.shape}")
            outputs.append(output)
            last_input = output
            attention = self.attention_func(torch.cat([states_down[-1][1], matched_memory], dim=2))
            memory_feature_att = matched_memory * attention
            tmp = torch.cat((states_down[-1][1], memory_feature_att), 2)
            states_down[-1][1] = self.fuse(tmp)

        return torch.stack(outputs, dim=1)

    def set_memory_bank_requires_grad(self, requires_grad):
        self.memory_bank.requires_grad=requires_grad


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.contiguous().view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.contiguous().view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.contiguous().view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.contiguous().view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.contiguous().view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=2, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.red = nn.Linear(2 * dim, dim)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.contiguous().view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, hx=None):
        # print(f"SwinTransformerBlock Input: {x.shape}, hx: {hx.shape if hx is not None else None}")
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)
        x = x.contiguous().view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.contiguous().view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.contiguous().view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            x = torch.roll(attn_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = attn_windows

        x = x.contiguous().view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # print(f"SwinTransformerBlock Output: {x.shape}")
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.contiguous().view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.contiguous().view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.contiguous().view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.contiguous().view(B, -1, C // 4)
        x = self.norm(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchInflated(nn.Module):
    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv_ = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=stride,
                               padding=padding, output_padding=output_padding),
            nn.GroupNorm(16, embed_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=64, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv_(x)
        x = self.Conv(x)

        return x


class SwinLSTMCell(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, depth, n_group=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        super(SwinLSTMCell, self).__init__()
        self.DATSwin = DATSwinTransformer(dim=dim, input_resolution=input_resolution, depth=depth,
                                          n_head=num_heads, window_size=window_size, n_group=n_group,
                                          mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                          norm_layer=norm_layer, downsample=None, use_checkpoint=False,
                                          use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False)

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)
        else:
            hx, cx = hidden_states

        Ft = self.DATSwin(xt, hx)
        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)
        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        hx = hy
        cx = cy
        return hx, [hx, cx]


class DATSwinTransformer(nn.Module):
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

        self.layers = nn.ModuleList([
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
                                    fixed_pe=fixed_pe)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, xt, hx):
        outputs = []
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint:
                if idx == 0:
                    x = checkpoint.checkpoint(layer, xt, hx)
                    outputs.append(x)
                else:
                    if idx % 2 == 0:
                        x = checkpoint.checkpoint(layer, outputs[-1], xt)
                        outputs.append(x)
                    if idx % 2 == 1:
                        x = checkpoint.checkpoint(layer, outputs[-1], None)
                        outputs.append(x)
            else:
                if idx == 0:
                    x = layer(xt, hx)
                    outputs.append(x)
                else:
                    if idx % 2 == 0:
                        x = layer(outputs[-1], xt)
                        outputs.append(x)
                    if idx % 2 == 1:
                        x = layer(outputs[-1], None)
                        outputs.append(x)
        return outputs[-1]


class SwinTransformer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        super(SwinTransformer, self).__init__()
        self.layers = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[depth - i - 1] if (flag == 0) else drop_path[i],
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, xt, hx):
        outputs = []
        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)
                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)
        return outputs[-1]


class DownSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super(DownSample, self).__init__()
        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]
        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                      dim=int(embed_dim * 2 ** i_layer))
            layer = SwinLSTMCell(dim=int(embed_dim * 2 ** i_layer),
                                 input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                   patches_resolution[1] // (2 ** i_layer)),
                                 depth=depths_downsample[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths_downsample[:i_layer]):sum(depths_downsample[:i_layer + 1])],
                                 norm_layer=norm_layer)
            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):
        x = self.patch_embed(x)
        # print(f"DownSample patch_embed: {x.shape}")
        hidden_states_down = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)
            # print(f"DownSample layer {index} output: {x.shape}, hidden_state: {hidden_state[0].shape}")
        return hidden_states_down, x


class UpSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_upsample, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, flag=0):
        super(UpSample, self).__init__()
        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.Unembed = PatchInflated(in_chans=embed_dim, embed_dim=embed_dim, input_resolution=patches_resolution)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]
        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_layers):
            resolution1 = (patches_resolution[0] // (2 ** (self.num_layers - i_layer)))
            resolution2 = (patches_resolution[1] // (2 ** (self.num_layers - i_layer)))
            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(resolution1, resolution2), dim=dimension)
            layer = SwinLSTMCell(dim=dimension, input_resolution=(resolution1, resolution2),
                                 depth=depths_upsample[(self.num_layers - 1 - i_layer)],
                                 num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths_upsample[:(self.num_layers - 1 - i_layer)]):
                                               sum(depths_upsample[:(self.num_layers - 1 - i_layer) + 1])],
                                 norm_layer=norm_layer, flag=flag)
            self.layers.append(layer)
            self.upsample.append(upsample)

        # 调整最终的上采样层，以确保输出尺寸为384x384
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=in_chans, kernel_size=1, stride=1, padding=0),  # 原来的是 64

            nn.Sigmoid()
        )

    def forward(self, x, y):
        hidden_states_up = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)
            # print(f"UpSample layer {index} output: {x.shape}, hidden_state: {hidden_state[0].shape}")

        x = torch.sigmoid(self.Unembed(x))
        # print(f"UpSample Unembed output: {x.shape}")
        # 通过新的上采样层将输出恢复到384x384
        B, C, H, W = x.shape  # 调整形状为B, C, H, W
        x = self.final_upsample(x)  # 恢复到384x384
        # print(f"UpSample final_upsample output: {x.shape}")
        return hidden_states_up, x





class SwinLSTM(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, depths_upsample, num_heads,
                 window_size):
        super(SwinLSTM, self).__init__()
        self.Downsample = DownSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                     embed_dim=embed_dim, depths_downsample=depths_downsample,
                                     num_heads=num_heads, window_size=window_size)
        self.Upsample = UpSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                 embed_dim=embed_dim, depths_upsample=depths_upsample,
                                 num_heads=num_heads, window_size=window_size)

    def forward(self, input, states_down, states_up):
        states_down, x = self.Downsample(input, states_down)
        states_up, output = self.Upsample(x, states_up)
        return output, states_down, states_up





if __name__ == '__main__':
    import os
    import argparse
    import torch

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parser = argparse.ArgumentParser('SwinLSTM testing script', add_help=False)
    # parser.add_argument('--input_img_size', default=384, type=int, help='Input image size')
    # parser.add_argument('--patch_size', default=4, type=int, help='Patch size of input images')  # 修改patch_size为8
    # parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    # parser.add_argument('--embed_dim', default=128, type=int,
    #                     help='Patch embedding dimension')  # 根据新的patch_size调整embed_dim
    # parser.add_argument('--depths_down', default=[4,4], type=int, help='Downsample of SwinLSTM-D')
    # parser.add_argument('--depths_up', default=[4,4], type=int, help='Upsample of SwinLSTM-D')
    # parser.add_argument('--heads_number', default=[4, 8], type=int,
    #                     help='Number of attention heads in different layers')
    # parser.add_argument('--window_size', default=4, type=int, help='Window size of Swin Transformer layer')
    # parser.add_argument('--short_len', default=12, type=int, help='Short length for memory encoder')
    # parser.add_argument('--long_len', default=36, type=int, help='Long length for memory context encoder')
    # parser.add_argument('--out_len', default=24, type=int, help='Output sequence length')

    # args = parser.parse_args()

    # model = Memory(args, memory_channel_size=512, short_len=args.short_len, long_len=args.long_len).to(device)
    # # print('\nModel is loaded!')

    # input_tensor = torch.randn(1, 12, 1, 384, 384).to(device)
    # memory_x = torch.randn(1, 24, 1, 384, 384).to(device)
    # phase = 2

    # with torch.no_grad():
    #     output = model(input_tensor, memory_x, phase)

    # print('Output shape:', output.shape)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser('SwinLSTM testing script', add_help=False)
    parser.add_argument('--input_img_size', default=768, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch size of input images')
    parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    parser.add_argument('--embed_dim', default=128, type=int, help='Patch embedding dimension')
    parser.add_argument('--depths_down', default=[1, 1], type=int, help='Downsample of SwinLSTM-D')
    parser.add_argument('--depths_up', default=[1, 1], type=int, help='Upsample of SwinLSTM-D')
    parser.add_argument('--heads_number', default=[4, 8], type=int, help='Number of attention heads in different layers')
    parser.add_argument('--window_size', default=4, type=int, help='Window size of Swin Transformer layer')
    parser.add_argument('--short_len', default=13, type=int, help='Short length for memory encoder')
    parser.add_argument('--long_len', default=25, type=int, help='Long length for memory context encoder')
    parser.add_argument('--out_len', default=12, type=int, help='Output sequence length')

    args = parser.parse_args()

    model = Memory(args, memory_channel_size=512, short_len=args.short_len, long_len=args.long_len).to(device)
    print('\nModel is loaded!')

    input_tensor = torch.randn(1, 13, 1, 768, 768).to(device)
    memory_x = torch.randn(1, 25, 1, 768, 768).to(device)
    phase = torch.tensor([2], dtype=torch.int).to(device)

    # Define a wrapper for the model to handle multiple inputs
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, input_tensor, memory_x, phase):
            return self.model(input_tensor, memory_x, phase.item())

    wrapped_model = ModelWrapper(model).to(device)

    # Calculate the number of parameters
    print("\nParameter count of the model:")
    params = parameter_count_table(wrapped_model)
    print(params)

    # Calculate FLOPs
    inputs = (input_tensor, memory_x, phase)
    flops = FlopCountAnalysis(wrapped_model, inputs)
    print("\nFLOPs of the model:")
    print(f"Total FLOPs: {flops.total()}")

    # Print model summary manually
    print("\nModel Summary:")
    print(wrapped_model)