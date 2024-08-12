# python -m torch.distributed.launch --nproc_per_node
import sys
sys.path.append('/data_8t/WSG/code/MetNow-main/models/DATSwinLSTM_D_Memory')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from pytorch_lightning.strategies import DDPStrategy
from typing import Tuple
import io
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from matplotlib import colors
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchvision import transforms
import PIL.Image
import math
import numpy as np
from config import cfg
import datetime
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import argparse
from pytorch_msssim import ssim

from sevir_dataloader import SEVIRDataLoader
from sevir_torch_wrap import SEVIRTorchDataset
from sevir import SEVIRSkillScore  # 导入预定义的CSI计算类

from models.DATSwinLSTM_D_Memory.DATSwinLSTM_D_Memory_small import Memory
from sevir_vis_seq import save_example_vis_results

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "code/MetNow-main/saved_models")
# CHECKPOINT_PATH = os.getcwd()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print("Device:", device)

# # Color map settings
# colev = ["#99DBEA", "#52A5D1", "#3753AD", "#80C505", "#52C10D",
#          "#35972A", "#FAE33B", "#EAB81E", "#F78C2C", "#E2331F",
#          "#992B27", "#471713", "#BC5CC2", "#975CC0"]
# cmap = colors.ListedColormap(colev)
# levels = np.arange(0, 71, 5)
# norm = colors.BoundaryNorm(levels, cmap.N)

class SEVIRLightningDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super(SEVIRLightningDataModule, self).__init__()
        self.seq_len = kwargs.get('seq_len', 49)
        self.sample_mode = kwargs.get('sample_mode', 'sequent')
        self.stride = kwargs.get('stride', 12)
        self.batch_size = kwargs.get('batch_size', 1)
        self.layout = kwargs.get('layout', 'NTCHW')
        self.output_type = kwargs.get('output_type', np.float32)
        self.preprocess = kwargs.get('preprocess', True)
        self.rescale_method = kwargs.get('rescale_method', '01')
        self.verbose = kwargs.get('verbose', False)
        self.num_workers = kwargs.get('num_workers', 0)
        dataset_name = kwargs.get('dataset_name', 'sevir')
        self.setup_dataset(dataset_name)
        self.start_date = datetime.datetime(*kwargs.get('start_date')) if kwargs.get('start_date') else None
        self.train_val_split_date = datetime.datetime(*kwargs.get('train_val_split_date', (2019, 1, 1)))
        self.train_test_split_date = datetime.datetime(*kwargs.get('train_test_split_date', (2019, 6, 1)))
        self.end_date = datetime.datetime(*kwargs.get('end_date')) if kwargs.get('end_date') else None

    def setup_dataset(self, dataset_name):
        if dataset_name == "sevir":
            sevir_root_dir = os.path.join(cfg.datasets_dir, "sevir")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevir_lr":
            sevir_root_dir = os.path.join(cfg.datasets_dir, "sevir_lr")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevir_lr'.")
        self.dataset_name = dataset_name
        self.sevir_root_dir = sevir_root_dir
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width

    def prepare_data(self) -> None:
        if os.path.exists(self.sevir_root_dir):
            assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            print('no data')

    def setup(self, stage=None) -> None:
        self.sevir_train = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=True,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.start_date,
            end_date=self.train_val_split_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        self.sevir_val = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.train_val_split_date,
            end_date=self.train_test_split_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        self.sevir_test = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        
        self.sevir_predict = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        
        print(f'Train set size: {len(self.sevir_train)}')
        print(f'Validation set size: {len(self.sevir_val)}')
        print(f'Test set size: {len(self.sevir_test)}')

    def train_dataloader(self):
        sampler = DistributedSampler(self.sevir_train) if torch.cuda.device_count() > 1 else None
        return DataLoader(self.sevir_train, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        sampler = DistributedSampler(self.sevir_val) if torch.cuda.device_count() > 1 else None
        return DataLoader(self.sevir_val, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        sampler = DistributedSampler(self.sevir_test) if torch.cuda.device_count() > 1 else None
        return DataLoader(self.sevir_test, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        sampler = DistributedSampler(self.sevir_predict) if torch.cuda.device_count() > 1 else None
        return DataLoader(self.sevir_predict, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)

class SwinLSTMLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = Memory(args, memory_channel_size=512, short_len=36, long_len=48)
        self.training_step_count = 0
        # self.csi_metric = SEVIRSkillScore(
        #     metrics_list=["csi"]
        # 
        self.automatic_optimization=False
        # self.automatic_optimization=True
        self.last_time = time.time()

    def forward(self, x, memory_x, phase):
        return self.model(x, memory_x, phase)
    
    def calculate_csi_pod(self, hits, misses, fas):
        csi = hits / (hits + misses + fas + 1e-10)
        pod = hits / (hits + misses + 1e-10)
        return csi, pod


    def training_step(self, batch, batch_idx):
        batch=batch.squeeze(0)
        loss_func = torch.nn.SmoothL1Loss(reduction="none")
        optimizer=self.optimizers()
        x, y = batch[:, :13, :, :, :], batch[:, 13:, :, :, :]
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        y_combined = batch[:,1:,:,:,:]

        # Phase 1
        self.model.set_memory_bank_requires_grad(True)
        memory_x = batch
        y_hat = self(x, memory_x, phase=1)
        if isinstance(y_hat,list):
            y_hat=torch.stack(y_hat)
        loss_phase_1 = loss_func(y_combined, y_hat)
        # loss_phase_1=F.l1_loss(y_combined, y_hat, reduction="none")
        loss_phase_1=loss_phase_1.mean()
        self.manual_backward(loss_phase_1)
        optimizer.step()
        optimizer.zero_grad()


        # Phase 2
        self.model.set_memory_bank_requires_grad(False)
        memory_x = x
        y_hat = self(x, memory_x, phase=2)
        if isinstance(y_hat, list):
            y_hat = torch.stack(y_hat)   
        loss_phase_2 = loss_func(y_combined, y_hat)     
        # loss_phase_2 = F.l1_loss(y_combined, y_hat, reduction="none")
        loss_phase_2 = loss_phase_2.mean()
        self.manual_backward(loss_phase_2)
        optimizer.step()
        optimizer.zero_grad()


        # Calculate and log FPS
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
            self.log('fps', fps)
            print(f"FPS: {fps:.2f}")
        self.last_time = current_time

        # 组合损失
        total_loss = loss_phase_1 + loss_phase_2
        # total_loss = loss_phase_2
        self.log('train_loss', total_loss)
        

        if self.training_step_count % 100 == 0:
            self.save_vis_step_end(x, y_combined, y_hat)

        # 创建局部 csi_metric 实例
        device = self.device
        csi_metric = SEVIRSkillScore(metrics_list=["csi"],dist_sync_on_step=False).to(device)
        csi_metric.update(y_combined, y_hat)
        # 确保 hits, fas 和 misses 都是 Tensor
        hits = csi_metric.hits.clone().detach().to(device)
        fas = csi_metric.fas.clone().detach().to(device)
        misses = csi_metric.misses.clone().detach().to(device)
        thresholds = csi_metric.threshold_list
        csi_values = []
        pod_values = []
        for i, threshold in enumerate(thresholds):
            csi, pod = self.calculate_csi_pod(hits[i], misses[i], fas[i])
            csi_values.append(csi)
            pod_values.append(pod)
            self.log(f'train_csi_{threshold}', csi)
            self.log(f'train_pod_{threshold}', pod)
        # 计算所有进程的平均值
        avg_csi = torch.mean(torch.tensor(csi_values))
        avg_pod = torch.mean(torch.tensor(pod_values))

        self.log('train_csi_avg', avg_csi)
        self.log('train_pod_avg', avg_pod)
        self.training_step_count = self.training_step_count+1
        # self.log("train_loss", loss_phase_2)
        # # 打印平均CSI和POD
        # print(f"Process {dist.get_rank()} avg CSI: {avg_csi} avg POD: {avg_pod}")
        return total_loss
        # return loss_phase_2

    def validation_step(self, batch, batch_idx):
        batch=batch.squeeze(0)
        x = batch[:, :13, :, :, :]
        y = batch[:, 13:, :, :, :]
        loss_func = torch.nn.SmoothL1Loss(reduction="none")
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        memory_x = x
        self.model.set_memory_bank_requires_grad(False)
        y_hat = self(x, memory_x, phase=2)
        
        y_combined = batch[:, 1:, :, :, :]
        if isinstance(y_hat, list):
            y_hat = torch.stack(y_hat)
        loss = loss_func(y_combined, y_hat)
        # loss = F.l1_loss(y_combined, y_hat, reduction="none")
        loss = loss.mean()
        self.log('val_loss', loss)
        # print("val loss:",loss)
        # csi = calculate_csi(y_combined, y_hat)
        # 创建局部 csi_metric 实例
        device = self.device
        csi_metric = SEVIRSkillScore(metrics_list=["csi"],dist_sync_on_step=False).to(device)
        csi_metric.update(y_combined, y_hat)
        # 确保 hits, fas 和 misses 都是 Tensor
        hits = csi_metric.hits.clone().detach().to(device)
        fas = csi_metric.fas.clone().detach().to(device)
        misses = csi_metric.misses.clone().detach().to(device)
        thresholds = csi_metric.threshold_list
        # 计算并记录不同阈值下的 CSI 和 POD
        csi_values = []
        pod_values = []
        for i, threshold in enumerate(thresholds):
            csi, pod = self.calculate_csi_pod(hits[i], misses[i], fas[i])
            csi_values.append(csi)
            pod_values.append(pod)
            self.log(f'val_csi_{threshold}', csi)
            self.log(f'val_pod_{threshold}', pod)
        avg_csi = torch.mean(torch.tensor(csi_values))
        avg_pod = torch.mean(torch.tensor(pod_values))
        self.log('val_csi_avg', avg_csi)
        self.log('val_pod_avg', avg_pod)
    
        # return loss

    def test_step(self, batch, batch_idx):
        batch=batch.squeeze(0)
        # optimizer=self.optimizers()
        x, y = batch[:, :13, :, :, :], batch[:, 13:, :, :, :]
        loss_func = torch.nn.SmoothL1Loss(reduction="none")
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        memory_x = x
        y_combined = batch[:, 1:, :, :, :]
        y_hat = self(x, memory_x, phase=2)
        
        # y_hat = y_hat[:, -24:, :, :, :]
        
        if isinstance(y, list):
            y = torch.stack(y)
        if isinstance(y_hat, list):
            y_hat = torch.stack(y_hat)

        loss = loss_func(y_combined, y_hat)    
        # loss = F.l1_loss(y, y_hat, reduction="none")
        loss = loss.mean()
        self.log('test_loss', loss)

        tar_seq = batch[:, -36:, :, : , :]
        pred_seq = y_hat[:, -36:, :, : , :]
        # 创建局部 csi_metric 实例
        device = self.device
        csi_metric = SEVIRSkillScore(metrics_list=["csi"],dist_sync_on_step=False).to(device)
        csi_metric.update(tar_seq, pred_seq)
        # 确保 hits, fas 和 misses 都是 Tensor
        hits = csi_metric.hits.clone().detach().to(device)
        fas = csi_metric.fas.clone().detach().to(device)
        misses = csi_metric.misses.clone().detach().to(device)
        thresholds = csi_metric.threshold_list
        csi_values = []
        pod_values = []
        for i, threshold in enumerate(thresholds):
            csi, pod = self.calculate_csi_pod(hits[i], misses[i], fas[i])
            csi_values.append(csi)
            pod_values.append(pod)
            self.log(f'test_csi_{threshold}', csi)
            self.log(f'test_pod_{threshold}', pod)
        avg_csi = torch.mean(torch.tensor(csi_values))
        avg_pod = torch.mean(torch.tensor(pod_values))
        self.log('test_csi_avg', avg_csi)
        self.log('test_pod_avg', avg_pod)


    def save_vis_step_end(
            self,
            in_seq: torch.Tensor,
            target_seq: torch.Tensor,
            pred_seq: torch.Tensor
    ):
        if self.local_rank==0:
            save_example_vis_results(                
                save_dir= os.path.join(CHECKPOINT_PATH, "swinlstm"),
                save_prefix=f'example_{self.training_step_count}',
                in_seq=in_seq.detach().float().cpu().numpy(),
                target_seq=target_seq.detach().float().cpu().numpy(),
                pred_seq=pred_seq.detach().float().cpu().numpy(),
                layout= 'NTCHW',
                plot_stride=2,
                label='DATSwinLSTM',
                interval_real_time=5)



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)# 需要优化的参数,lr学习率 Adam优化器，自适应调整lr
        #min：指标不在减少时，减小lr；factor：减少时的乘的因子；patience：多少epoch不改善减少lr；min_lr：lr下限
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}



def get_train_images(dm, num):
    return dm.sevir_train[num]


def train_sevir():
    # Initialize the process group
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
    
    # Get local rank from the environment variable set by torchrun or torch.distributed.launch
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    print(f'Process {local_rank} is using device {device}')

    dm = SEVIRLightningDataModule()
    dm.setup()
    input_img = get_train_images(dm, 0)
    print(input_img.shape)

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    print(f"Total training steps per epoch: {len(train_dataloader)}")
    print(f"Total validation steps per epoch: {len(val_dataloader)}")

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "swinlstm"),
        accelerator='gpu',
        devices=4,  # 使用4张GPU卡
        strategy=DDPStrategy(find_unused_parameters=True),  # 启用DDP策略
        max_epochs=200,
        precision=32,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, monitor='val_csi_16', save_top_k=3, mode='max'),
            LearningRateMonitor("epoch"),
            # GenerateCallback(input_img, every_n_epochs=1),
            EarlyStopping(monitor='val_csi_16', mode='max', patience=30),
        ],
        log_every_n_steps=1,
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    args = argparse.Namespace(
        input_img_size=384,
        patch_size=4,
        input_channels=1,
        embed_dim=128,
        depths_down=[1,1],
        depths_up=[1,1],
        heads_number=[4, 8],
        window_size=4,
        out_len=36
    )

    model = SwinLSTMLightningModule(args).to(device)


    print(f'Train set size: {len(dm.sevir_train)}')
    if len(dm.sevir_train) > 0:
        sample = dm.sevir_train[0]
        print(f'Sample train data shape: {sample.shape}')

    trainer.fit(model, train_dataloader, val_dataloader)
    val_result = trainer.test(model, dataloaders=val_dataloader, verbose=False)
    test_result = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
    result = {"test": test_result, "val": val_result}

    # Clean up the process group
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

    return model, result



def set_device_for_distributed():
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    return device

if __name__ == '__main__':
    device = set_device_for_distributed()
    print(f"Running on device: {device}")
    model, result = train_sevir()
    print(result)

