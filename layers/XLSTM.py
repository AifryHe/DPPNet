from einops import rearrange
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

import argparse

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat, pack, unpack
from layers.Autoformer_EncDec import series_decomp

mlstm_config = mLSTMBlockConfig()
slstm_config = sLSTMBlockConfig()


class XLSTM(nn.Module):

    def __init__(self, configs) -> None:
        super(XLSTM, self).__init__()
        # config_s = xLSTMBlockStackConfig(
        #     mlstm_block=mlstm_config,
        #     slstm_block=slstm_config,
        #     num_blocks=3,
        #     embedding_dim=256,
        #     add_post_blocks_norm=True,
        #     # _block_map=1,
        #     slstm_at="all",
        #     context_length=862 if enc_in == 862 else 336,
        # )
        config_m = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            # slstm_block=slstm_config,
            num_blocks=1,
            embedding_dim=256,
            add_post_blocks_norm=True,
            _block_map=1,
            # slstm_at="all",
            context_length=862,
        )

        self.enc_in = configs.enc_in
        self.context_points = configs.seq_len
        self.target_points = configs.pred_len
        self.embedding_dim = configs.n2

        # self.weighting_after = WeightingLayer(self.target_points)
        # self.weighting_before = WeightingLayer(self.embedding_dim)
        self.batch_norm = nn.BatchNorm1d(self.enc_in)

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.context_points, self.context_points)
        self.Linear_Trend = nn.Linear(self.context_points, self.context_points)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.context_points)
            * torch.ones([self.context_points, self.context_points])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.context_points)
            * torch.ones([self.context_points, self.context_points])
        )

        self.mm = nn.Linear(self.context_points, self.embedding_dim)
        self.mm2 = nn.Linear(self.embedding_dim, self.target_points)
        self.activation = nn.GELU()
        # self.mm3 = nn.Linear(self.context_points, self.n2)
        # if xlstm_kind == "s":
        #     self.xlstm_stack = xLSTMBlockStack(config_s)
        # else:
        self.xlstm_stack = xLSTMBlockStack(config_m)

    def forward(self, x):
        # print(x.shape) #x(batch_size, num, seq_len)
        # batch time variate
        # x = x.permute(0, 2, 1) #(B, M, L)
        seasonal_init, trend_init = self.decomposition(x.permute(0, 2, 1))
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        x = self.mm(x)
        x = self.batch_norm(x)
        x = self.activation(x)  # 激活函数添加在第一个线性层之后

        x = self.xlstm_stack(x)
        x = self.activation(x)  # 激活函数添加在 XLSTM 层之后
        x = self.mm2(x)

        # x = rearrange(x, "b v n  -> b n v")

        return x

# if __name__ == '__main__':
#     # 设置随机种子
#     torch.manual_seed(0)
#     np.random.seed(0)
#
#     # 定义模型参数
#     pred_len = 24  # 预测长度
#     seq_len = 48  # 输入序列长度
#     enc_in = 10  # 特征数量（变量数量）
#     batch_size = 16  # 批次大小
#
#     # 创建模型实例
#     model = XLSTM(pred_len=pred_len, seq_len=seq_len, enc_in=enc_in)
#
#     # 创建示例输入数据
#     x_enc = torch.randn(batch_size, seq_len, enc_in)
#     x_mark_enc = torch.randn(batch_size, seq_len, enc_in)  # 如果模型中未使用，可以省略或传入占位符
#
#     # 将数据移动到模型所在的设备（如 GPU）
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     x_enc = x_enc.to(device)
#     x_mark_enc = x_mark_enc.to(device)
#
#     # 前向传播
#     output = model(x_enc, x_mark_enc)
#
#     # 输出结果的形状
#     print("输出形状：", output.shape)
#     # 应该为 (batch_size, pred_len, enc_in)，即 (16, 24, 10)

