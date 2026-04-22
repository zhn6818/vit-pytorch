import torch  # 导入 PyTorch 主库，用于创建张量和运行模型
import sys  # 导入系统模块，用于修改模块搜索路径
sys.path.append("./")  # 将当前目录添加到 Python 模块搜索路径，确保能找到 vit_pytorch 包

from vit_pytorch import ViT  # 从 vit_pytorch 包中导入 ViT 模型类

def test_vit():  # 定义测试函数，用于验证 ViT 模型的基本功能
    v = ViT(  # 实例化 ViT 模型
        image_size = 256,  # 输入图像尺寸为 256×256 像素
        patch_size = 32,  # 每个图像块大小为 32×32 像素，即将图像切成 8×8=64 个块
        num_classes = 1000,  # 分类数为 1000（如 ImageNet 的 1000 类）
        dim = 1024,  # token 嵌入维度为 1024
        depth = 6,  # Transformer 编码器堆叠 6 层
        heads = 16,  # 多头注意力使用 16 个头
        mlp_dim = 2048,  # 前馈网络的隐藏层维度为 2048
        dropout = 0.1,  # Transformer 内部的丢弃率为 10%
        emb_dropout = 0.1  # 嵌入后的丢弃率为 10%
    )

    img = torch.randn(1, 3, 256, 256)  # 生成随机输入张量，形状为 (1, 3, 256, 256)，即 1 张 3 通道 256×256 的图像

    preds = v(img)  # 将图像输入 ViT 模型，得到分类预测 logits
    print(preds.shape)  # 打印输出张量的形状，预期为 (1, 1000)
    assert preds.shape == (1, 1000), 'correct logits outputted'  # 断言输出形状为 (1, 1000)，否则报错

if __name__ == '__main__':  # 当直接运行此脚本时（而非被导入时）
    test_vit()  # 执行测试函数
