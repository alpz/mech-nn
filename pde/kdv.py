
import torch.nn as nn
import torch

from solver.ode_layer import ODESYSLayer
from torch.nn.parameter import Parameter
import numpy as np

import torch
import ipdb
import torch.optim as optim
import pytorch_lightning as pl
import h5py

from scipy.special import logit
import os

from torch.utils.data import Dataset

from extras.source import write_source_files, create_log_dir
import sys
from tqdm import tqdm
import pde.resnet1d as R


DBL = False
learning_rate = 0.0001
n_time_steps = 10
batch_size = 32

eval_freq=5

class KDVDataset(Dataset):
    def __init__(self, kind='train'):

        self.steps_per_example = 10
        data_root = 'datasets/KDV_easy'
        if kind=='train':
            file = os.path.join(data_root,'KdV_train_512_easy.h5')
            h5_path = 'train/pde_140-256'
            #self.n_time_steps = 140
            #self.n_time_steps = 40
        elif kind=='valid':
            file = os.path.join(data_root,'KdV_valid_easy.h5')
            h5_path = 'valid/pde_140-256'
            #self.n_time_steps = 640
        elif kind=='test':
            file = os.path.join(data_root,'KdV_test_easy.h5')
            h5_path = 'test/pde_140-256'
            #self.n_time_steps = 640
        else:
            raise ValueError('Invalid dataset type') 

        f = h5py.File(file, 'r')
        data = f.get(h5_path)
        data = np.array(data)
        self.data = torch.tensor(data, dtype=torch.float64 if DBL else torch.float32)
        self.n_trajectory = self.data.shape[0]
        self.n_time_steps = self.data.shape[1]
        print('data module ', kind, self.data.shape, self.n_time_steps)
        self.steps_per_index = 1
        self.length_per_trajectory= self.n_time_steps-self.steps_per_example+1-9

        
    def __len__(self):
        length = self.length_per_trajectory*self.n_trajectory
        return length
    def __getitem__(self, idx):
        traj_idx = idx//self.length_per_trajectory
        time_idx = idx - traj_idx*self.length_per_trajectory
        time_idx = time_idx*self.steps_per_index

        x = self.data[traj_idx, time_idx:time_idx+10]
        y = self.data[traj_idx, time_idx+9:time_idx+9+self.steps_per_example]
        return x,y
    
class KDVDataModule(pl.LightningDataModule):
    def __init__(self ):
        super().__init__()
        self.train_dataset = KDVDataset(kind='train')
        self.valid_dataset = KDVDataset(kind='valid')
        self.test_dataset = KDVDataset(kind='test')

        self.batch_size = batch_size

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=9)
        return train_loader 

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size,drop_last=True, shuffle=False, num_workers=9)
        return loader 

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=9)
        return loader 
        
        

class Method(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = KDV(device=self.device)
        if DBL:
            self.model = self.model.double()

    def setup(self, stage):
        if stage=='fit':
            print('Source files ', self.logger.log_dir)
            write_source_files(self.logger.log_dir)
        
    def forward(self, *args):
        return self.model(*args, check=False)

    def loss(self, x,y):
        mae = (x-y).abs().sum(dim=-1)

        return mae.mean()

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        u,eps = self(x,y)
        

        loss = self.loss(u,y) 

        
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        #self.log('max eps', eps, prog_bar=True, logger=True, on_step=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        u,eps = self(x,y)
        

        loss = self.loss(u,y) 

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True)
        
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        #eps, u0 = self(x,y)
        u,eps = self(x,y)
        
        loss = self.loss(u,y) 
        
        self.log('test_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, , 80, 130, 130, 180, 230, 260], gamma=0.4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        #return [optimizer], [scheduler]
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": eval_freq,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

def get_losses(method, datamodule):
    losses = []
    for i in range(0,512, batch_size):
        x = datamodule.test_dataset.data[i:i+batch_size]
        loss = generate_traj_losses(x, method, datamodule)
        losses.extend(loss)
    losses = torch.cat(losses, dim=0).mean()
    return losses.item()

def generate_traj_losses(data, method: Method, datamodule: KDVDataModule):
    method.eval()

    true_trajectory = data.cuda()
    x = true_trajectory[:, :n_time_steps]


    losses = []

    out_list =[] # [x.unsqueeze(1)]
    n_step=14
    for i in tqdm(range(n_step)):
        u, u_x,_,_ = method(x, None, None)
        
        u = torch.cat([x[:,:-1], u[:,:]], dim=1)
        x = torch.cat([x[:,:-1], u[:,:]], dim=1)
        u_x = u[:, :-n_time_steps]
        x = u[:, -n_time_steps:]
        out_list.append(u_x)

    prediction = torch.cat(out_list, dim=1)
    true_x = true_trajectory[:, n_time_steps:100+n_time_steps]
    prediction = prediction[:, n_time_steps:100+n_time_steps]

    loss  = (prediction - true_x).pow(2).mean(dim=-1)
    d = true_x.pow(2).mean(dim=-1)
    loss = loss/d
    loss = loss.mean(dim=1)

    losses.append(loss.squeeze())

    return losses

def generate_traj(method: Method, datamodule: KDVDataModule):
    method.eval()
    n_step = datamodule.test_dataset.length_per_trajectory

    data =[]
    for i in range(batch_size):
        data.append(datamodule.test_dataset.data[i])

    true_trajectory = torch.stack(data, axis=0).cuda()
    print(true_trajectory.shape)
    x = true_trajectory[:, :10]

    x = x.cuda()

    out_list =[] # [x.unsqueeze(1)]
    n_step=140
    iv_t = None
    for i in tqdm(range(n_step)):
        u, u_x,_,_ = method(x, None, None)
        
        u = torch.cat([x[:,:-1], u[:,:]], dim=1)
        x = torch.cat([x[:,:-1], u[:,:]], dim=1)
        #u =(u +x)/2
        u_x = u[:, :-10]
        x = u[:, -10:]
        out_list.append(u_x)
        

    trajectory = torch.cat(out_list, dim=1)
    print(trajectory.shape)

    return trajectory.detach().cpu().numpy(), true_trajectory.cpu().numpy()



class KDV(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.step_size_t = 0.05
        self.order = 4
        #state dimension
        self.n_dim = 1
        self.bs = batch_size
        #kself.n_coeff = self.n_step * (self.order + 1)
        self.device  = device
        dtype = torch.float64

        self.coord_dims = (n_time_steps,256)
        self.n_iv = 1


        self.n_ind_dim=256
        self.n_step = self.coord_dims[0]
        self.n_dim=1 #self.coord_dims[1]//self.n_ind_dim
        self.n_equations=self.n_dim
        #self.n_step_ = self.coord_dims[0]
        self.n_iv=self.order
        self.n_iv_steps=1
        self.ode = ODESYSLayer(bs=self.bs, n_ind_dim=self.n_ind_dim, order=self.order, n_equations=self.n_equations, gamma=0.05,alpha=0.,
                                     n_dim=self.n_dim, n_iv=self.n_iv, n_step=self.n_step, n_iv_steps=self.n_iv_steps, solver_dbl=True, double_ret=False)


        pm = 'circular'


        self._cf_cnn = R.ResNet1D(in_channels=10, base_filters=64, kernel_size=9, stride=1, groups=1, n_block=10, n_classes=2, use_bn=False, use_do=False)
        self.cf_cnn = nn.Sequential(
            nn.Conv1d(256,(self.order+1)*self.n_step, kernel_size=7, padding=3, stride=1, padding_mode=pm),
            
            )

        self.step_cnn = nn.Sequential(
            nn.Conv1d(10,64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=5, padding=2, stride=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,256, kernel_size=5, padding=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,256, kernel_size=5, padding=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,64, kernel_size=5, padding=2, stride=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,32, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            #nn.Conv1d(64,7*self.coord_dims[0], kernel_size=5, padding=2),
            nn.Flatten()
            )


        self._rhs_cnn = R.ResNet1D(in_channels=10, base_filters=64, kernel_size=9, stride=1, groups=1, n_block=10, n_classes=2, use_bn=False, use_do=False)
        self.rhs_cnn = nn.Sequential(
            nn.Conv1d(256,self.n_step, kernel_size=7, padding=3, stride=1, padding_mode=pm),
            )

        self.iv_cnn = nn.Sequential(
            nn.Conv1d(10,64, kernel_size=7, padding=3, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=7, padding=3, stride=1, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,256, kernel_size=7, padding=3, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,256, kernel_size=7, padding=3, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(256,128, kernel_size=7, padding=3, stride=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(128,64, kernel_size=7, padding=3, stride=2, padding_mode=pm),
            nn.ReLU(),
            nn.Conv1d(64,32, kernel_size=7, padding=3, stride=1, padding_mode=pm),
            #nn.Conv1d(64,7*self.coord_dims[0], kernel_size=5, padding=2),
            nn.Flatten()
            )

        self.ode_iv_out  = nn.Linear(32*64,self.n_ind_dim*(self.order-1))

        self.steps_layer_t = nn.Linear(32*64, 1) #self.coord_dims[1])

        #set step bias to set initial step
        step_bias_t = logit(self.step_size_t)

        self.steps_layer_t.weight.data.fill_(0.0)
        self.steps_layer_t.bias.data.fill_(step_bias_t)


    def forward(self, x, y, _iv_t=None, check=False):
        __coeffs = self._cf_cnn(x)
        __coeffs = self.cf_cnn(__coeffs)
        rhs_coeffs = self._rhs_cnn(x)
        rhs_coeffs = self.rhs_cnn(rhs_coeffs)
        iv_coeffs = self.iv_cnn(x)
        step_coeffs = self.step_cnn(x)

        _coeffs = __coeffs
        _rhs = rhs_coeffs
        _iv = self.ode_iv_out(iv_coeffs)

        _coeffs = _coeffs.reshape(-1,  self.coord_dims[0]*(self.order+1), self.coord_dims[1]).permute(0,2,1)
        _rhs = _rhs.reshape(-1, self.coord_dims[0], self.coord_dims[1]).permute(0,2,1)

        coeffs = _coeffs
        rhs = _rhs


        if _iv_t is None:
            iv_t = x[:,-1:,:].reshape(-1,1,self.n_ind_dim,self.n_dim).permute(0,2,1,3)

            iv_t = iv_t.reshape(-1, self.coord_dims[1],1)
            _iv = _iv.reshape(-1, self.coord_dims[1],self.order-1)

            iv_t = torch.cat([iv_t, _iv], dim=-1)


        steps_t = self.steps_layer_t(step_coeffs).repeat(1, self.coord_dims[1], self.coord_dims[0]-1)

        steps_t = torch.sigmoid(steps_t).clip(min=0.001)# max=0.3)

        #################
        u,_,_,eps,u_t_all = self.ode(coeffs, rhs, iv_t, steps_t)

        u = u.reshape(x.shape[0], self.n_ind_dim, self.n_step, self.n_dim)
        u = u.permute(0,2,1,3).reshape(-1,self.n_step, self.coord_dims[1])

        u_t_all = u_t_all.reshape(x.shape[0], self.coord_dims[1], self.coord_dims[0], self.order+1)

        return u,eps.abs().max()

datamodule = KDVDataModule()


trainer = pl.Trainer(
    max_epochs=800,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #accelerator="cpu",
    check_val_every_n_epoch=eval_freq,
    devices=1,
    callbacks=[
        pl.callbacks.ModelCheckpoint(mode="min", monitor="val_loss"),
    ],
    log_every_n_steps=1,
)

def train():
    method = Method()
    trainer.fit(method, datamodule=datamodule)

def load(ckpt):
    method = Method.load_from_checkpoint(ckpt)
    #trainer.test(method, datamodule=datamodule)
    with torch.no_grad():
        return generate_traj(method, datamodule)

def losses(ckpt):
    method = Method.load_from_checkpoint(ckpt)
    #trainer.test(method, datamodule=datamodule)
    with torch.no_grad():
        return get_losses(method, datamodule)


def resume(ckpt):
    method = Method.load_from_checkpoint(ckpt)
    trainer.fit(method, datamodule=datamodule)

if __name__ == "__main__":
    if len(sys.argv) ==1:
        print('training')
        train()
    else:
        losses(sys.argv[1])
    #load(sys.argv[1])
    #resume(sys.argv[1])
