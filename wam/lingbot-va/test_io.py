import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.configs import va_a2d_train_cfg # 引入你的配置

# 👇 就是加了这极其关键的一行！保护主进程逻辑不被子进程乱跑！
if __name__ == '__main__':
    print("正在初始化 Dataset...")
    dset = MultiLatentLeRobotDataset(va_a2d_train_cfg)
    dloader = DataLoader(dset, batch_size=va_a2d_train_cfg.batch_size, shuffle=True, num_workers=16)

    print("开始测试纯数据读取速度 (不涉及任何 GPU 计算)...")
    start_time = time.time()

    # 只空转读取 20 个 Batch
    for i, data in enumerate(dloader):
        if i >= 20:
            break

    total_time = time.time() - start_time
    print(f"\n✅ 测试完成！")
    print(f"纯 Dataloader 读取 1 个 Batch 的平均耗时: {total_time / 20:.2f} 秒")