import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
from tqdm import tqdm

# from src.models import resnet, vgg19
from src.resnet import resnet50, resnet101, resnet18

from src.losses.ot_loss import OT_Loss
from src.utils.pytorch_utils import Save_Handle, AverageMeter
import src.utils.log_utils as log_utils
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter



def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[2])
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return images, points, st_sizes, gt_discretes


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args

        self.writer = SummaryWriter('/dm_count/logs/' + args.experiment)

        sub_dir = args.experiment

        self.save_dir = os.path.join('outputs', sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8
        self.datasets = {}

        if args.train_dataset == 'blendercam':
            from src.datasets.crowd import Crowd_blendercam

            self.datasets['train'] = Crowd_blendercam((args.train_dir),
                                        args.crop_size, downsample_ratio, 'train')
        
        if args.val_dataset == 'blendercam':
            from src.datasets.crowd import Crowd_blendercam

            self.datasets['val'] = Crowd_blendercam((args.train_dir),
                                        args.crop_size, downsample_ratio, 'val')
        if args.train_dataset == 'blendercam_preprocess':
            from src.datasets.crowd import Crowd_blendercam_preprocess

            self.datasets['train'] = Crowd_blendercam_preprocess((args.train_dir),
                                        args.crop_size, downsample_ratio, 'train')
        
        if args.val_dataset == 'blendercam_preprocess':
            from src.datasets.crowd import Crowd_blendercam_preprocess

            self.datasets['val'] = Crowd_blendercam_preprocess((args.train_dir),
                                        args.crop_size, downsample_ratio, 'val')

        if args.train_dataset == 'people':
            from src.datasets.crowd import Crowd_sh

            self.datasets['train'] = Crowd_sh((args.val_dir),
                                        args.crop_size, downsample_ratio, 'train')

        if args.val_dataset == 'sht-b':
            from src.datasets.crowd import Crowd_sh

            self.datasets['val'] = Crowd_sh((args.val_dir),
                                        args.crop_size, downsample_ratio, 'train') #!!

        if args.train_dataset == 'penguins':
            from src.datasets.crowd import Crowd_penguins

            self.datasets['train'] = Crowd_penguins((args.train_dir),
                                        args.crop_size, args.train_limit, downsample_ratio, 'train')
        
        if args.val_dataset == 'penguins':
            from src.datasets.crowd import Crowd_penguins

            self.datasets['val'] = Crowd_penguins((args.val_dir),
                                        args.crop_size, args.val_limit, downsample_ratio, 'val')
        
        if args.train_dataset == 'vehicles':
            from src.datasets.crowd import Crowd_trancos

            self.datasets['train'] = Crowd_trancos((args.train_dir),
                                        args.crop_size, args.train_limit, downsample_ratio, 'training')
        
        if args.val_dataset == 'vehicles':
            from src.datasets.crowd import Crowd_trancos

            self.datasets['val'] = Crowd_trancos((args.val_dir),
                                        args.crop_size, args.val_limit, downsample_ratio, 'validation')
        
        if args.train_dataset == 'apples':
            from src.datasets.crowd import Crowd_apples

            self.datasets['train'] = Crowd_apples((args.train_dir),
                                        args.crop_size, downsample_ratio, 'train')
        
        if args.val_dataset == 'apples':
            from src.datasets.crowd import Crowd_apples

            self.datasets['val'] = Crowd_apples((args.val_dir),
                                        args.crop_size, downsample_ratio, 'test')

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          worker_init_fn=np.random.seed(args.seed),
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        if args.model == 'resnet50':
            self.model = resnet50(pretrained = True)
        elif args.model == 'resnet18':
            self.model = resnet18(pretrained = True)
        elif args.model == 'resnet50-f2':
            self.model = resnet50(pretrained = True)
            for param in self.model.layer1.parameters():
                param.requires_grad = False
            for param in self.model.layer2.parameters():
                param.requires_grad = False


        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=args.lr, max_lr=args.max_lr, step_size_up=7, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

        self.start_epoch = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')

        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot,
                               args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()

        under5_errors_distribution = []
        error_contribution_tmp = [0] * 4000
        error_contribution = []
        under5_successes = 0

        self.model.train()  # Set model to training mode

        for step, (inputs, points, st_sizes, gt_discrete) in enumerate(tqdm(self.dataloaders['train'])):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs)
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gd_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

                l_outputs = outputs.sum(1).sum(1).sum(1).int().tolist()
                print('----')
                print(l_outputs)
                l_labels = torch.from_numpy(gd_count).int().tolist()
                print(l_labels)

                for x in range(0, len(l_labels)):
                    if l_outputs[x] != l_labels[x]:
                        error_contribution_tmp[l_labels[x]] += abs(l_outputs[x] - l_labels[x])
                        
                    if abs(l_outputs[x] - l_labels[x]) < 5:
                        under5_successes += 1
                    else:
                        under5_errors_distribution.append(int(l_labels[x]))
        
        for i, c in enumerate(error_contribution_tmp):
            error_contribution += [i] * c

        self.scheduler.step()

        self.logger.info(
            'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
            'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                .format(self.epoch, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
                        epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
                        np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                        time.time() - epoch_start))
        
        self.writer.add_scalar('Counting MAE/train', epoch_mae.get_avg(), self.epoch)
        self.writer.add_scalar('Under 5 errors/train', np.asarray(under5_successes), self.epoch)
        self.writer.add_histogram('Under 5 errors distribution/train', np.asarray(under5_errors_distribution), self.epoch)
        self.writer.add_histogram('Error contribution/train', np.asarray(error_contribution), self.epoch)

        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        under5_errors_distribution = []
        error_contribution_tmp = [0] * 4000
        error_contribution = []
        under5_successes = 0

        for inputs, count, name in tqdm(self.dataloaders['val']):
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, _ = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                if res != 0:
                    error_contribution_tmp[count[0].item()] += abs(int(res))
                    
                if abs(res) < 5:
                    under5_successes += 1
                else: 
                    under5_errors_distribution.append(int(count[0].item()))

                epoch_res.append(res)

        for i, c in enumerate(error_contribution_tmp):
            error_contribution += [i] * c

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        self.writer.add_scalar('Counting MAE/val', mae, self.epoch)
        self.writer.add_scalar('Under 5 errors/val', np.asarray(under5_successes), self.epoch)
        self.writer.add_histogram('Under 5 errors distribution/val', np.asarray(under5_errors_distribution), self.epoch)
        self.writer.add_histogram('Error contribution/val', np.asarray(error_contribution), self.epoch)


        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1