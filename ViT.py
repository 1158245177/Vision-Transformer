import math

import torch
import torch.nn as nn  # 导入神经网络
from einops import rearrange, repeat  # 图像切割重构


#把一个图像分割成patches，变为一维向量的类
class PatchEmbedding(nn.Module):  #继承nn.Module
    def __init__(self, x, patch_size=16, dim=768):
        super().__init__()  #调用父类构造函数
        batch, channel, image_width, image_height = x.shape
        #判断patch_size能够被image_width整除
        assert image_width % patch_size == 0 and image_height % patch_size == 0, "Image size must be divided by the patch size"
        self.patch_size = patch_size
        num_patches = (image_height * image_height) // (patch_size * patch_size)
        patch_dim = patch_size * patch_size * channel

        self.to_patch_embedding = nn.Linear(patch_dim, dim, bias=False)  #将拉直后的patch_dim 输入到一个全连接层中，变成dim。不设置偏置
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  #CLASS token 是一个可训练的参数，经过Linear层 故维度为dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  #每一个patch需要一个位置编码

    def forward(self, x):
        b, c, w, h = x.shape
        #（1，3，224，224）→1，（14，14），（16，16，3）代表一共14×14个patches，每个patch大小16×16×3
        x = rearrange(x, 'b c (w p1) (h p2)->b (w h) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x1 = self.to_patch_embedding(x)

        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)  #原始的cls进行重复，为了更好的训练
        x2 = torch.cat([self.cls_token, x1], dim=1)  #将图像patches转换的向量和cls token进行拼接
        x2 = x2 + self.pos_embedding
        return x2


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        #定义q,k,v
        self.query = nn.Linear(dim, dim, bias=False)  #一维定义为Linear 若为2维定义为矩阵matrix
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.dim)
        attention_score = nn.Softmax(dim=1)(attention_score)
        out = torch.bmm(attention_score, V)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, dropout=0.1, project_out=False):  #project_out是否需要线性映射
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        # 定义q,k,v
        self.query = nn.Linear(dim, dim, bias=False)  # 一维定义为Linear 若为2维定义为矩阵matrix
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(  #是否需要线性映射
            nn.Linear(dim, dim),  #输入输出维度均为dim，可以改动维度
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  #如果不需要，则无需做任何动作

    def forward(self, x):
        Q = rearrange(self.query(x), 'b n (h d) ->b h n d', h=self.num_heads)
        K = rearrange(self.key(x), 'b n (h d) ->b h n d', h=self.num_heads)
        V = rearrange(self.value(x), 'b n (h d) ->b h n d', h=self.num_heads)

        attention_score = torch.einsum('b h q d,b h k d -> b h q k', Q, K)  #矩阵相乘，q,d and k,d ->q,k#多头里面K不用转置？
        attention_score = nn.Softmax(dim=1)(attention_score) / math.sqrt(self.dim)
        out = torch.einsum('b h a n,b h n v -> b h a v', attention_score, V)
        out = rearrange(out, 'b h n d -> b n (h d)')  #把维度变化回来
        return self.to_out(out)


class FeedForeardBlock(nn.Sequential):  #前馈层，继承了Sequential
    def __init__(self, dim=768, expansion=4, dropout=0.1):
        super().__init__(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * dim, dim)  #映射回来
        )


class TransformerEncoder(nn.Module):
    def __init__(self, dim=768):  #定义一个构造函数
        super().__init__()
        self.LN = nn.LayerNorm(dim)  #LinearNorm
        self.MHA = MultiHeadAttention()  #多头注意力
        self.FFB = FeedForeardBlock()  #前馈层

    def forward(self, x):
        x1 = self.LN(x)
        x2 = self.MHA(x1)
        x2 = x2 + x  #残差xiang'jia
        x3 = self.LN(x2)
        x4 = self.FFB(x3)
        x4 = x4 + x2

        return x4


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    patch_embedding = PatchEmbedding(x)  #先是一个embedding
    x1 = patch_embedding(x)
    print(x1.shape)
    transformer = TransformerEncoder()  # 然后是一个Transformer Encoder
    x2 = transformer(x1)
    for _ in range(5):
        x2 = transformer(x2)
    print(x2.shape)
