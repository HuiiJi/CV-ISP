import os
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from utils.model_utils import init_all_random
from utils.dir_utils import mkdirs
from utils.image_utils import reduce_mean, cal_score, torchPSNR, numpyPSNR, save_gated_png
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import get_training_data,  get_validation_data
import argparse
import matplotlib.pyplot as plt
import torchvision


def train_(args):
    init_all_random(args.local_rank)
    ######### Model ###########
    Network_list = {

    }
    model = Network_list[args.model_]
    ######### mutilply GPU  ###########
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    model = DDP(model.cuda(), device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    ######### criterion list + opimizer ###########
    criterion_list = {
        'L1': torch.nn.L1Loss('mean').to(device),
    }
    criterion = criterion_list[args.criterion]
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-8,
                           weight_decay=args.weight_decay,
                           )
    ######### Scheduler-warmup+cosine ###########
    if args.warm_up:
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)
    ######### trainvaloader  ###########
    train_dataset = get_training_data(args.train_dir, {"patch_size": args.train_ps})
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler = train_sampler, shuffle=False, num_workers=args.num_worker, drop_last=True, pin_memory=True)
    val_dataset = get_validation_data(args.train_dir, {'patch_size':args.train_ps})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    ######### train_stage ###########
    try:
        ckp = torch.load(args.ckp, map_location = 'cpu')
        model.load_state_dict(ckp['state_dict'])
        print('--------------------------train stage-------------------------')
    except:
        print('-----------------------------load faild-----------------------------')

    best_loss = 666
    best_psnr = 0.01
    EPOCH_LOSS = []

    for epoch_idx in range(1, args.num_epoch):
        model.train()
        train_sampler.set_epoch(epoch_idx)
        loss_list = []
        for iter_idx, batch_data in tqdm(enumerate(train_loader)):
            label_img, input_img= batch_data

            input_img = input_img.cuda(args.local_rank, non_blocking=True)
            label_img = label_img.cuda(args.local_rank, non_blocking=True)

      
            restored, gated = model(input_img)
                
            loss = criterion(restored, label_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = reduce_mean(loss, dist.get_world_size())
            loss_list.append(loss.item())

        scheduler.step()
        ave_loss = sum(loss_list) / len(loss_list)
        EPOCH_LOSS.append(ave_loss)

        if rank == 0:
            if best_loss >= ave_loss:
                torch.save({'epoch': epoch_idx, 
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(args.model_save_dir, args.model_ + "best.pth"))
                # print('Epoch = %d' % epoch_idx, '----best_loss = %.6f' % best_loss,
                #         'last_loss = %.6f----' % ave_loss, 'lr = %.6f' % optimizer.param_groups[0]['lr'])
                best_loss = ave_loss
            else:
                torch.save({'epoch': epoch_idx, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(args.model_save_dir, args.model_ + str(epoch_idx) + '.pth'))
                
            
            plt.plot(EPOCH_LOSS, linewidth=1)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('training loss curve')
            plt.savefig(os.path.join(args.result_dir, 'loss_curve.png'))
            plt.close()
          
            if (int(epoch_idx)) % args.val_every == 0:
                model.eval()
                post_psnr_list = []
                post_ssim_list = []
                psnr_ori_list= []
                ssim_ori_list =[]
                AVE_PSNR = []
           
                with torch.no_grad():
                    for idx, data in enumerate(val_loader):
                        label_img, input_img = data
                        input_img = input_img.to(device)
                        label_img = label_img.to(device)

                        if 'GatedDIP_' in args.model_:
                            restored, gated = model(input_img)
                        elif 'IAT_' in args.model_:
                            restored = model(input_img)[2]
                        else:
                            restored, gated = model(input_img)
                        
                        # torchvision.utils.save_image(torch.cat([restored, input_img, label_img], 3), os.path.join(args.result_dir, str(idx) + '_cvisp_out.png'))      
                        torchvision.utils.save_image(restored, os.path.join(args.result_dir, str(epoch_idx) + str(idx) + '_cvisp_out.png'))
                        save_gated_png(gated, os.path.join(args.result_dir, str(epoch_idx) + str(idx) + '_gated.png'))
                        
                        result_data = restored.cpu().detach().numpy().transpose(0, 2, 3, 1)
                        input_ = input_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
                        gt = label_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
                        
                        post_psnr = numpyPSNR(gt, result_data)
                        orig_psnr = numpyPSNR(gt, input_)
                        change_psnr = post_psnr - orig_psnr
                        # change_ssim = post_ssim - orig_ssim
                        post_psnr_list.append(post_psnr)
                        # post_ssim_list.append(post_ssim)
                        psnr_ori_list.append(orig_psnr)

                ave_post_psnr = sum(post_psnr_list) / len(post_psnr_list)
                # ave_post_ssim = sum(post_ssim_list) / len(post_ssim_list)
                AVE_PSNR.append(ave_post_psnr)
                ave_orig_psnr = sum(psnr_ori_list) / len( psnr_ori_list)
                if best_psnr < ave_post_psnr:
                    best_psnr = ave_post_psnr
                # ave_orig_ssim = sum(ssim_ori_list) / len( ssim_ori_list)
                change_ave_psnr = ave_post_psnr - ave_orig_psnr
                # change_ave_ssim = ave_post_ssim - ave_orig_ssim
                # plt.plot(epoch_idx, AVE_PSNR, linewidth=1)
                # plt.xlabel('Epoch')
                # plt.ylabel('%')
                # plt.title('PSNR on validation')
                # plt.savefig(os.path.join(args.result_dir, 'PSNR_curve.png'))
                print('Epoch%03d----Best_PSNR %.4f---orig_psnr ---%.4f---current_psnr---%.4f' % (epoch_idx, best_psnr, ave_orig_psnr, ave_post_psnr))
                print(gated)
    dist.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--normalize', action='store_true', help='Default False')
    parser.add_argument('--train_dir', type=str, default = '') 
    parser.add_argument('--learning_rate', type=float, default=2e-4) 
    parser.add_argument('--weight_decay', type=float, default=2e-4)  ##
    parser.add_argument('--train_ps', type=int, default=512)  
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--criterion', type=str, default='L1')
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--warm_up', type=bool, default=False)
    args = parser.parse_args()
    torch.cuda.empty_cache()
    args.model_save_dir = os.path.join('', args.model_)
    args.result_dir = os.path.join('', args.model_)

    mkdirs([args.model_save_dir, args.result_dir])
    show_txt = open(os.path.join(args.result_dir, 'train_log.txt'), 'w')
    show_txt.write(str(args) + '\n')

    train_(args)



