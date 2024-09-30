# 用于将monitor生成的文件保存为event files文件
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import wandb
from model import  Model
from Taiyi.taiyi.monitor import Monitor
from Taiyi.visualize import Visualization

def prepare_data(w, h, class_num, len):
    x = torch.randn((len, 3, w, h), requires_grad=True)
    y = torch.randint(0, class_num, (len,))
    return x, y


def prepare_optimizer(model, lr=1e-2):
    return torch.optim.SGD(model.parameters(), lr=lr)


def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_taiyi = {
        nn.Conv2d: ['InputSndNorm']
        #nn.BatchNorm2d: [['MeanTID', 'linear(5,0)'],'InputSndNorm']
    }
    return config, config_taiyi


def prepare_loss_func():
    return nn.CrossEntropyLoss()


if __name__ == '__main__':
    config, config_taiyi = prepare_config()
    x, y = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
    model = Model(config['w'], config['h'], config['class_num'])
    opt = prepare_optimizer(model, config['lr'])
    loss_fun = prepare_loss_func()
    #######################################
    monitor = Monitor(model, config_taiyi)
    vis = Visualization(monitor,project=config_taiyi.keys(),name=config_taiyi.values())
    writer = SummaryWriter(log_dir='logs')
    #######################################
    for epoch in range(config['epoch']):
        opt.zero_grad() # 清除梯度的函数
        y_hat = model(x)
        loss = loss_fun(y_hat, y)
        loss.backward()
        ##############################
        monitor.track(epoch)
        logs =  vis.show(epoch)
        for key,value in logs.items():
            print(f"当前保存{key}，其对应的值为{value}")
            writer.add_scalar(key,value,epoch)
        print(logs)
        print(epoch)
        #########################
        # #####
        opt.step() # 参数更新
    print(monitor.get_output())