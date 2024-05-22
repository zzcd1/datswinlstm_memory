import argparse
from torch.nn import Conv2d, ConvTranspose2d


def get_args():
    parser = argparse.ArgumentParser('zzwei code for precipitation nowcasting', add_help=False)
    # Setup parameters
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--res_dir', default='/home/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/zzwei_paper_code/configs/zzwei_results/', type=str)
    parser.add_argument('--seed', default=1234, type=int)    
    parser.add_argument('--dataset', default='SEVIR', type=str, choices=['SEVIR','SRAD2018','Shanghai2020','CIKM2017'])
    parser.add_argument('--accumulation_steps',default=2,type=int)
    parser.add_argument('--use_ema',default=True,type=bool)
    parser.add_argument('--resume',default=False,type=bool)
    parser.add_argument('--weight_loss',default=False,type=bool)
    parser.add_argument('--freq_loss',default=False,type=bool)
    parser.add_argument('--ema_decay', type=float,
                        default=0.9999, help='decay rate for EMA')  
    # model parameters
    parser.add_argument('--model_name', default='DATSWinLSTM-D-Memory', type=str, choices=['MotionRNN','rainymotion','DATSWinLSTM-D-Memory','DATSWinLSTM-B-Memory','SwinLSTM-B', 'SwinLSTM-D','DAT-SWIN','MIMO-VP','Rainformer','SimVP','DAT-SWIN-z','ConvLSTM','EN_PredRNN','TrajGRU','PFST_LSTM'],
                        help='Model type')
    parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    parser.add_argument('--input_img_size', default=384, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch size of input images for swinlstm')
    parser.add_argument('--embed_dim', default=128, type=int, help='Patch embedding dimension')
    parser.add_argument('--depths', default=[12], type=int, help='Depth of Swin Transformer layer for SwinLSTM-B')
    parser.add_argument('--depths_down', default=[3, 2], type=int, help='Downsample of SwinLSTM-D')
    parser.add_argument('--depths_up', default=[2,3], type=int, help='Upsample of SwinLSTM-D')
    parser.add_argument('--heads_number', default=[4,8], type=int,
                        help='Number of attention heads in different layers')
    parser.add_argument('--window_size', default=4, type=int, help='Window size of Swin Transformer layer')
    parser.add_argument('--drop_rate', default=0., type=float, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='Attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')
    # 模型参数
    ## ============== encoder_forecaster ================
    parser.add_argument('--encode_cells',type=list,default=[[8,64,3,3,3],[64,192,3,3,3],[192,192,3,3,3]])
    parser.add_argument('--encode_padding',type=list,default=[[3],[2],[1]])
    parser.add_argument('--decode_cells',type=list,default=[[192,192,3,3,3],[192,192,3,3,3],[192,64,3,3,3]])
    parser.add_argument('--decode_padding',type=list,default=[[1,1],[1,0],[2,1]])
    parser.add_argument('--deshape',type=list,default=[16,32,96,384])
    parser.add_argument('--downsampling_convs',type=list,default=[[1,8,7,4],[64,64,5,3],[192,192,3,2]])
    parser.add_argument('--upsample_convs',type=list,default=[[192,192,3,2],[192,192,5,3],[64,8,7,4]])
    parser.add_argument('--output_conv',type=list,default=[[8,1,1,1]])
    parser.add_argument('--output_padding',type=list,default=[[0]])

    parser.add_argument('--m_encode',type=list,default=[[64,64,3,3],[64,64,3,3]])
    parser.add_argument('--m_encode_padding',type=list,default=[[1,1],[1,2]])
    parser.add_argument('--m_decode',type=list,default=[[64,64,3,3],[64,64,3,3]])
    parser.add_argument('--m_decode_padding',type=list,default=[[1,1],[1,2]])
    parser.add_argument('--enshape',type=list,default=[384,96,32,16])
    parser.add_argument('--m_channels',type=list,default=[64,64,64])  

    ## TrajGRU的参数
    parser.add_argument('--Ls',type=list,default=[13,13,9])
    ## PFST_LSTM
    parser.add_argument('--input_cell',type=list,default=[[1,8,1,1],[8,192,7,4],[192,192,5,3],[192,192,3,2]])
    parser.add_argument('--input_padding',type=list,default=[[0],[3],[2],[1]])    
    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training,validation and test')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--epoch_valid', default=1, type=int)
    parser.add_argument('--epoch_test', default=1, type=int)
    parser.add_argument('--log_train', default=200, type=int)
    parser.add_argument('--log_valid', default=100, type=int)
    parser.add_argument('--lr', default=0.00012, type=float, help='Learning rate') # shanghai2020波动大
    parser.add_argument('--lr_d',default=0.0001)
    parser.add_argument('--early_stop_gan',default=1000)
    parser.add_argument('--test_result_dir',default='/home/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/zzwei_paper_code/tmp_1/')#'/root/data/WeatherRadar/test_results/SRAD/trajgru_gan/')#'/root/data/WeatherRadar/test_results/SEVIR/paper/datswinlstm_memory_gan')#'/root/ljy_space/zzw/zzwei_results/Shanghai2020-ConvLSTM/pred_gan/')#'/root/data/WeatherRadar/test_results/SEVIR/simvp_gan/')#'/root/ljy_space/zzw/zzwei_results/cikm2017_results/MotionRNN/')#'/root/ljy_space/zzw/zzwei_results/SRAD2018-DATSWIN-LSTM-D-Memory/pred/') # /root/ljy_space/zzw/zzwei_results/SEVIR-DATSWinLSTM-D-Memory/pred/'/home/ljy_space/zzw/srad_test_results_with_z/'    
    
    args = parser.parse_args()
    
    if args.dataset == 'SRAD2018':
        args.train_data_dir = '/root/data/WeatherRadar/srad/train/'
        args.validation_data_dir='/root/data/WeatherRadar/srad/val/'
        args.test_data_dir = '/root/data/WeatherRadar/srad/test/'
        args.total_length = 24
        args.num_frames_input = 12
        args.num_frames_output = 12
        args.input_size = (384,384)
        args.short_len = 12
        args.long_len = 24
        args.out_len = 12  
    elif args.dataset == 'SEVIR':
        args.train_data_dir = '/home/ljy/Desktop/8T_mount/SEVIR/processed_3(12_24)(copy_1)/train/'
        args.validation_data_dir='//home/ljy/Desktop/8T_mount/SEVIR/processed_3(12_24)(copy_1)/validation/'
        args.test_data_dir = '/home/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/dat_swin_code_prepare/seq/' #'/home/ljy/Desktop/8T_mount/SEVIR/processed_3(12_24)(copy_1)/test/'
        args.total_length = 24
        args.num_frames_input = 12
        args.num_frames_output = 12
        args.input_size = (384,384)
        args.short_len = 12
        args.long_len = 24
        args.out_len = 24           
    elif args.dataset == 'CIKM2017':
        args.train_data_dir = '/root/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/CIKM2017_dataset/train/'
        args.validation_data_dir='/root/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/CIKM2017_dataset/validation/'
        args.test_data_dir = '/root/zhaozewei/Documents/WeatherRadar/RadarExtrapolation/CIKM2017_dataset/test/'
        args.total_length = 15
        args.num_frames_input = 5
        args.num_frames_output = 10
        args.input_size = (128,128)
        args.input_img_size=128

        args.short_len = 5
        args.long_len = 15
        args.out_len = 10                
    elif args.dataset == 'Shanghai2020':
        print(f'dataset:{args.dataset} path is written in the dataloader file!')
        args.total_length=20
        args.num_frames_input=10
        args.num_frames_output=10
        args.input_size=(384,384)
        args.short_len=10
        args.long_len=20
        args.out_len=10     
    ## MIMO-VP args
    if args.model_name == 'MIMO-VP':
        #
        args.d_model=128
        args.n_layers=6
        args.heads=8
        args.dropout=0
    if args.model_name == 'MotionRNN':
        args.LSTM_conv = Conv2d
        args.LSTM_deconv = ConvTranspose2d
        args.CONV_conv = Conv2d
        args.lstm_hidden_state = 128#64 ## 64 ## 66
        args.kernel_size = 3
        args.LSTM_layers = 3
        args.CONV_conv = Conv2d
        args.use_scheduled_sampling = True        
    print(args)    
    return args
