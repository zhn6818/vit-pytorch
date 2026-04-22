import torch  # 导入 PyTorch 主库，提供张量运算等基础功能
from torch import nn  # 从 PyTorch 中导入神经网络模块，用于构建网络层
from torch.nn import Module, ModuleList  # 导入 Module 基类和 ModuleList 容器，用于定义可训练的自定义模块

from einops import rearrange, repeat  # 导入 einops 库的张量重排和重复函数，用于简洁的张量维度操作
from einops.layers.torch import Rearrange  # 导入 einops 的 PyTorch 层封装，可作为 nn.Module 使用

# helpers  辅助函数

def pair(t):
    # 如果 t 已经是元组则直接返回，否则将其转为 (t, t) 元组
    # 用于统一处理 image_size 和 patch_size 可能为 int 或 (int, int) 的情况
    return t if isinstance(t, tuple) else (t, t)

# classes  类定义

class FeedForward(Module):  # 定义前馈神经网络（MLP），继承自 Module
    def __init__(self, dim, hidden_dim, dropout = 0.):  # dim 为输入/输出维度，hidden_dim 为隐藏层维度，dropout 为丢弃率
        super().__init__()  # 调用父类 Module 的初始化方法
        self.net = nn.Sequential(  # 使用 Sequential 容器按顺序堆叠各层
            nn.LayerNorm(dim),  # 对输入做层归一化，稳定训练
            nn.Linear(dim, hidden_dim),  # 线性层：将维度从 dim 映射到 hidden_dim（升维）
            nn.GELU(),  # GELU 激活函数，提供非线性变换（ViT 中常用）
            nn.Dropout(dropout),  # 随机丢弃部分神经元，防止过拟合
            nn.Linear(hidden_dim, dim),  # 线性层：将维度从 hidden_dim 映射回 dim（降维）
            nn.Dropout(dropout)  # 再次随机丢弃，进一步正则化
        )

    def forward(self, x):  # 前向传播，x 为输入张量
        return self.net(x)  # 将输入依次通过上述所有层并返回结果


class Attention(Module):  # 定义多头自注意力机制，继承自 Module
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):  # dim 为输入维度，heads 为注意力头数，dim_head 为每个头的维度，dropout 为丢弃率
        super().__init__()  # 调用父类初始化
        inner_dim = dim_head *  heads  # 计算所有头的总维度 = 每头维度 × 头数
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要输出投影：仅当单头且维度一致时不需要

        self.heads = heads  # 保存注意力头数
        self.scale = dim_head ** -0.5  # 缩放因子 = 1/√dim_head，用于防止点积值过大

        self.norm = nn.LayerNorm(dim)  # 注意力前的层归一化（Pre-LN 结构）

        self.attend = nn.Softmax(dim = -1)  # Softmax 函数，沿最后一维（即 key 维度）计算注意力权重
        self.dropout = nn.Dropout(dropout)  # 对注意力权重进行随机丢弃

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 线性层：将输入投影为 Q、K、V 三部分（拼接在一起），不使用偏置

        self.to_out = nn.Sequential(  # 输出投影层：将多头拼接的结果映射回原始维度
            nn.Linear(inner_dim, dim),  # 线性层：从 inner_dim 映射回 dim
            nn.Dropout(dropout)  # 随机丢弃
        ) if project_out else nn.Identity()  # 如果不需要投影，则使用恒等映射

    def forward(self, x):  # 前向传播，x 形状为 (batch, seq_len, dim)
        x = self.norm(x)  # 先对输入做层归一化（Pre-LN）

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 通过线性投影得到 QKV，然后沿最后一维切成 3 份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 将每份重排为多头格式：(batch, seq_len, inner_dim) -> (batch, heads, seq_len, dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算注意力分数：Q × K^T，再乘以缩放因子，形状为 (batch, heads, seq_len, seq_len)

        attn = self.attend(dots)  # 对注意力分数做 Softmax，得到注意力权重矩阵
        attn = self.dropout(attn)  # 对注意力权重做随机丢弃

        out = torch.matmul(attn, v)  # 用注意力权重对 V 加权求和，得到注意力输出，形状为 (batch, heads, seq_len, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 将多头输出拼接回去：(batch, heads, seq_len, dim_head) -> (batch, seq_len, inner_dim)
        return self.to_out(out)  # 通过输出投影层（或恒等映射）返回最终结果


class Transformer(Module):  # 定义 Transformer 编码器，由多个注意力 + 前馈层堆叠而成
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):  # dim 为特征维度，depth 为堆叠层数，heads 为头数，dim_head 为每头维度，mlp_dim 为前馈层隐藏维度，dropout 为丢弃率
        super().__init__()  # 调用父类初始化
        self.norm = nn.LayerNorm(dim)  # 最终的层归一化，用于在所有层结束后做归一化
        self.layers = ModuleList([])  # 用 ModuleList 存储所有 Transformer 层（不会被当作普通列表忽略）

        for _ in range(depth):  # 循环 depth 次，创建 depth 个 Transformer 块
            self.layers.append(ModuleList([  # 每个块包含一个注意力层和一个前馈层
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),  # 多头自注意力层
                FeedForward(dim, mlp_dim, dropout = dropout)  # 前馈神经网络层
            ]))

    def forward(self, x):  # 前向传播，x 形状为 (batch, seq_len, dim)
        for attn, ff in self.layers:  # 遍历每个 Transformer 块
            x = attn(x) + x  # 注意力层输出 + 残差连接（将注意力输出加回输入）
            x = ff(x) + x  # 前馈层输出 + 残差连接（将前馈输出加回输入）

        return self.norm(x)  # 对最终输出做层归一化后返回


class ViT(Module):  # 定义 Vision Transformer（ViT）模型，继承自 Module
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):  # 使用仅关键字参数；image_size 为图像尺寸，patch_size 为块大小，num_classes 为分类数，dim 为嵌入维度，depth 为 Transformer 深度，heads 为头数，mlp_dim 为前馈隐藏维度，pool 为池化方式，channels 为图像通道数，dim_head 为每头维度，dropout 和 emb_dropout 为丢弃率
        super().__init__()  # 调用父类初始化
        image_height, image_width = pair(image_size)  # 将图像尺寸统一为 (高, 宽) 元组
        patch_height, patch_width = pair(patch_size)  # 将块大小统一为 (高, 宽) 元组

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'  # 断言图像尺寸能被块大小整除

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 计算图像被切成的总块数 = (高/块高) × (宽/块宽)
        patch_dim = channels * patch_height * patch_width  # 计算每个块展平后的维度 = 通道数 × 块高 × 块宽

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 断言池化方式只能是 'cls' 或 'mean'
        num_cls_tokens = 1 if pool == 'cls' else 0  # 如果用 cls token 池化则添加 1 个 CLS token，否则为 0

        self.to_patch_embedding = nn.Sequential(  # 图像块嵌入层：将图像块转换为嵌入向量
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),  # 将图像切分为块并展平：(batch, channels, height, width) -> (batch, num_patches, patch_dim)
            nn.LayerNorm(patch_dim),  # 对展平后的块做层归一化
            nn.Linear(patch_dim, dim),  # 线性层：将块维度映射到嵌入维度 dim
            nn.LayerNorm(dim),  # 对嵌入向量再做一次层归一化
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))  # 可学习的 CLS token 参数，形状为 (num_cls_tokens, dim)，用于聚合全局信息
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))  # 可学习的位置编码参数，形状为 (序列长度, dim)，为每个 token 添加位置信息

        self.dropout = nn.Dropout(emb_dropout)  # 嵌入后的随机丢弃层

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer 编码器实例

        self.pool = pool  # 保存池化方式
        self.to_latent = nn.Identity()  # 恒等映射层（占位，方便后续扩展为额外映射）

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None  # 如果有分类任务则创建线性分类头，否则设为 None

    def forward(self, img):  # 前向传播，img 形状为 (batch, channels, height, width)
        batch = img.shape[0]  # 获取批次大小
        x = self.to_patch_embedding(img)  # 将图像切分并嵌入为 token 序列，形状为 (batch, num_patches, dim)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)  # 将 CLS token 沿批次维度复制，形状变为 (batch, 1, dim)
        x = torch.cat((cls_tokens, x), dim = 1)  # 将 CLS token 拼接到 token 序列的最前面，形状变为 (batch, num_patches+1, dim)

        seq = x.shape[1]  # 获取当前序列长度（包括 CLS token）

        x = x + self.pos_embedding[:seq]  # 加上位置编码（前 seq 个），为每个 token 注入位置信息
        x = self.dropout(x)  # 对嵌入后的序列做随机丢弃

        x = self.transformer(x)  # 通过 Transformer 编码器处理

        if self.mlp_head is None:  # 如果没有分类头（如仅用作特征提取器）
            return x  # 直接返回 Transformer 的输出

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 如果用 mean 池化则对所有 token 求均值，否则取 CLS token（第 0 个位置）

        x = self.to_latent(x)  # 通过恒等映射层（不做任何变换，可扩展）
        return self.mlp_head(x)  # 通过分类头输出分类 logits，形状为 (batch, num_classes)
