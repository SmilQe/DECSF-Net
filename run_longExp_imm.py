import argparse
import os
import torch

import time
from exp.exp_main_imm import Exp_Main_MM
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer && MultiModel Transformer for Time Series Forcasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='Random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                        help='model name, options: [Transformer, MultiModel Transformer]')

    # dataset and dataloader
    parser.add_argument('--data', type=str, required=True, default='water', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [M, S, MS]; M: multivariate predict multivariate, S: univariate predict univariate, MS: mutlivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--water_seq_len', type=int, default=336, help='water input sequence length')
    parser.add_argument('--mete_seq_len', type=int, default=432, help='mete input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Formers
    parser.add_argument('--water_enc_in', type=int, default=4, help='water encoder input size')
    parser.add_argument('--mete_enc_in', type=int, default=10, help='mete encoder input size')
    parser.add_argument('--water_dec_in', type=int, default=4, help='water decoder input size')
    parser.add_argument('--c_out', type=int, default=4, help='model output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--ew_layers', type=int, default=2, help='number of water encoder layers')
    parser.add_argument('--em_layers', type=int, default=2, help='number of mete encoder layers')
    parser.add_argument('--dw_layers', type=int, default=1, help='number of water decoder layers')
    parser.add_argument('--b_layers', type=int, default=1, help='number of bma attn layers in mm blocks')
    parser.add_argument('--ba_layers', type=int, default=1, help='number of attn layers in mm block ')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options: [timeF, fixes, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader use num workers to load data')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='use single gpu to train model, gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multi_gpu')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tool for usage')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.device = args.devices.replace(" ", " ")  # 清除 args.device 中的空格
        device_ids = args.device.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main_MM

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = (f'{args.model_id}_{args.model}_{args.data}_ft{args.features}'
                       f'_wsl{args.water_seq_len}_msl{args.mete_seq_len}_ll{args.label_len}_wpl{args.pred_len}_'
                       f'dm{args.d_model}_nh{args.n_heads}_wel{args.ew_layers}_mel{args.em_layers}_dwl{args.dw_layers}_'
                       f'bl{args.b_layers}_ba{args.ba_layers}_df{args.d_ff}_dp{args.dropout}_ac_{args.activation}_itr_{ii}_bs_{args.batch_size}_amp_{args.use_amp}')

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

    else:
        ii = 0
        # setting record of experiments
        setting = (f'{args.model_id}_{args.model}_{args.data}_ft{args.features}'
                   f'_wsl{args.water_seq_len}_msl{args.mete_seq_len}_ll{args.label_len}_wpl{args.pred_len}_'
                   f'dm{args.d_model}_nh{args.n_heads}_wel{args.ew_layers}_mel{args.em_layers}_dwl{args.dw_layers}_'
                   f'bl{args.b_layers}_ba{args.ba_layers}_df{args.d_ff}_dp{args.dropout}_ac_{args.activation}_itr_{ii}_bs_{args.batch_size}')

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        time_now = time.time()
        exp.test(setting, test=1)
        print(f"{args.water_seq_len}_{args.mete_seq_len}_{args.pred_len}_time: {time.time() - time_now} s")
        torch.cuda.empty_cache()
