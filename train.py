import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import time
import argparse
import torch.optim as optim
from lossfuntion import *
from dataset import ValDataset
from dataloaders import train_dali_loader
from utils import close_logger, init_logging, normalize_augment, ADDNOISE_class, loss_calc
from train_common import resume_training, log_train_psnr, validate_and_log, save_model_checkpoint
import numpy as np
import random
from network.MDIVDnet import MDIVDnet

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(**args):

    #set seed
    seed=args['seed']
    if seed == -1:
        seed = np.random.randint(1, 10000)
    set_seed(seed)

    if args['gpus_num'] == -1:
        device_ids = range(torch.cuda.device_count())
    else:
        device_ids = range(args['gpus_num'])
    device = torch.device('cuda' if device_ids[0] >= 0 else 'cpu')

    # Load dataset
    if args['valset_dir'] != '':
        print('> Loading datasets ...')
        dataset_val = ValDataset(valsetdir=args['valset_dir'], num_input_frames=args['framenum_of_val'])

    # define graymode
    GRAYmode = args['graymode']

    print("load train")
    loader_train = train_dali_loader(batch_size=args['batch_size'],
                                     file_root=args['trainset_dir'],
                                     sequence_length=args['temp_patch_size'],
                                     crop_size=args['patch_size'],
                                     epoch_size=args['max_number_patches'],
                                     random_shuffle=True,
                                     temp_stride=3)


    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    # Init loggers
    writer, logger, TIMESTAMP = init_logging(args)

    # define noise
    noise = ADDNOISE_class(image_size=(256, 320))

    # define model
    model = MDIVDnet()

    # Create model Parallel
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model = model.to(device)

    # Define loss
    criterion_L2 = L2_LOSS().cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    #set CosineRestart learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args['epochs'], T_mult=2,
                                                                         eta_min=1e-6)

    # Resume training or start anew
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Training
    start_time = time.time()
    for epoch in range(start_epoch, args['epochs']):

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        scheduler.step()
        print('\nlearning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):
            model.train()
            optimizer.zero_grad()
            img_train = normalize_augment(data[0]['data'])
            N, FC, H, W = img_train.size()
            if GRAYmode == True:
                C = 1
            else:
                C = 3
            F = int(FC/C)

            # add noise
            imgn_train = noise.addnoise(img_train, args, mode='train')
            imgn_train = imgn_train.cuda(non_blocking=True)

            # Evaluate model and optimize it
            out_train = model(imgn_train.reshape(N, F, C, H, W))
            out_train = out_train.reshape(N, FC, H, W)

            # Compute loss
            loss = loss_calc(args, criterion_L2, img_train, out_train)

            loss.backward()
            optimizer.step()

            # Results
            if training_params['step'] % args['save_every'] == 0:
                log_train_psnr(out_train,
                               img_train,
                               loss,
                               writer,
                               epoch,
                               i,
                               num_minibatches,
                               training_params)

            # update step counter
            training_params['step'] += 1

        # Validation and log images
        if args['valset_dir'] != '':
            psnr_val = validate_and_log(
                model_temp=model,
                dataset_val=dataset_val,
                args=args,
                writer=writer,
                epoch=epoch,
                lr=current_lr,
                training_params = training_params)
            # save model and checkpoint
            training_params['start_epoch'] = epoch + 1
            if epoch == start_epoch:
                best_info = [0, 0] # best epoch, best psnr
            psnr_val_up_flag, best_info = save_model_checkpoint(model, args, optimizer, training_params, epoch, TIMESTAMP, best_info, psnr_val)
            if psnr_val_up_flag == 1:
                if epoch == 0:
                    logger.info("Val PSNR: ")
                max_psnr_epoch = best_info[0]
                max_psnr = best_info[1]
            logger.info("epoch{}: {}, max_psnr_epoch: {}, max_psnr: {}".format(epoch+1, psnr_val, max_psnr_epoch, max_psnr))
        else:
            save_model_checkpoint(model, args, optimizer, training_params, epoch, TIMESTAMP)
            logger.info("epoch{} Save Model".format(epoch + 1))

        # Estimated remaining time
        Est_remaining_time = (time.time() - start_time) / (epoch - start_epoch + 1) * (args['epochs'] - (epoch + 1))
        day = int(time.strftime("%d", time.gmtime(Est_remaining_time)))-1
        print('Est_remaining_time: {:0>2d}:{}'.format(day, time.strftime("%H:%M:%S", time.gmtime(Est_remaining_time))))
    # Print elapsed time
    elapsed_time = time.time() - start_time
    day = int(time.strftime("%d", time.gmtime(elapsed_time))) - 1
    print('Elapsed time: {:0>2d}:{}'.format(day, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    print(TIMESTAMP)

    # Close logger file
    close_logger(logger)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the denoiser")

    #Training parameters
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=30,
                        help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true',
                        help="resume training from a previous checkpoint")
    parser.add_argument("--log_dir_resume", type=str, default="",
                        help="log_dir_resume")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--graymode", action='store_false')
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--auto_save_every", type=int, default=50)
    parser.add_argument("--save_every_epochs", type=int, default=5,
                        help="Number of training epochs to save state")
    parser.add_argument("--save_all", default="perfect",
                        help="save each epoch training model")
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed')
    parser.add_argument('--gpus_num', type=int, default=-1,
                        help='the number of gpus wanted to use')
    parser.add_argument("--framenum_of_val", type=int, default=15,
                        help="Number of frame in val sequence")

    parser.add_argument('--note', type=str, default="", help='the note which is wanted to be log in logs')
    parser.add_argument('--noteindir', type=str, default="", help='the note which is wanted to be named in dirs')

    # noise param
    # train
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 35],
                        help="GaussianNoise training interval")
    parser.add_argument("--noise_line_ival", nargs=2, type=int, default=[5, 35],
                        help="LineNoise training interval")
    parser.add_argument("--noise_rowspatial_ival", nargs=2, type=int, default=[0, 15],
                        help="row Noise training interval")
    parser.add_argument("--noise_colspatial_ival", nargs=2, type=int, default=[0, 15],
                        help="col Noise training interval")
    parser.add_argument("--noise_bias_ival", nargs=2, type=int, default=[5, 35],
                        help="bias_field Noise training interval")

    # val
    parser.add_argument("--val_noiseL", type=float, default=30,
                        help='noise level used on validation set')
    parser.add_argument("--val_line_noiseL", type=float, default=30,
                        help='noise level used on validation set')
    parser.add_argument("--val_row_noiseL", type=float, default=10,
                        help='noise level used on validation set')
    parser.add_argument("--val_col_noiseL", type=float, default=10,
                        help='noise level used on validation set')
    parser.add_argument("--val_bias_noiseL", type=float, default=30,
                        help='noise level used on validation set')

    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=256, help="Patch size")
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
    parser.add_argument("--max_number_patches", "--m", type=int, default=32000,
                        help="Maximum number of patches")

    # Dirs
    parser.add_argument("--log_dir", type=str, default="logs",
                        help='path of log files')
    parser.add_argument("--trainset_dir", type=str, default="./data/train_MP4",
                        help='path of trainset')
    parser.add_argument("--valset_dir", type=str, default="", help='path of validation set')
    argspar = parser.parse_args()

    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.val_line_noiseL /= 255.
    argspar.val_row_noiseL /= 255.
    argspar.val_col_noiseL /= 255.
    argspar.val_bias_noiseL /= 255.

    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.
    argspar.noise_line_ival[0] /= 255.
    argspar.noise_line_ival[1] /= 255.
    argspar.noise_rowspatial_ival[0] /= 255.
    argspar.noise_rowspatial_ival[1] /= 255.
    argspar.noise_colspatial_ival[0] /= 255.
    argspar.noise_colspatial_ival[1] /= 255.
    argspar.noise_bias_ival[0] /= 255.
    argspar.noise_bias_ival[1] /= 255.

    num_minibatches = int(argspar.max_number_patches // argspar.batch_size)
    if argspar.auto_save_every != 0:
        save_every_temp = num_minibatches // argspar.auto_save_every // 10
        if save_every_temp != 0:
            argspar.save_every = num_minibatches // argspar.auto_save_every // 10 * 10
        else:
            argspar.save_every = argspar.auto_save_every


    print("\n### Training FastDVDnet denoiser model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
