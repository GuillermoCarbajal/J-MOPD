import torch
from skimage.io import imread
from skimage.transform import rescale
from utils.homographies import compute_intrinsics, generate_video, get_offsets_from_positions, \
                                     save_kernels_from_offsets, show_kernels_from_offsets_on_blurry_image, save_motion_flow, reblur_offsets
import numpy as np
#from models.network_nimbusr_pmbm import NIMBUSR_PMBM as net
from models.network_nimbusr_offsets import NIMBUSR_Offsets as net_nimbusr_offsets

from utils.visualization import save_image, tensor2im, save_video, sort_positions, show_positions_found
import os 
import argparse
from models.CameraShakeModel_OffsetNet import OffsetNet_quad as OffsetsNet
import json
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--blurry_image', '-b', type=str, help='blurry image', default='./testing_imgs/manmade_01_gyro_01.png')
parser.add_argument('--reblur_model', '-m', type=str, help='reblur model', required=True)
parser.add_argument('--restoration_network', '-rn', type=str, help='restoration network', default=r'NIMBUSR/model_zoo/PMPB_220000_G.pth')
parser.add_argument('--rescale_factor','-rf', type=float, default=1)
parser.add_argument('--restoration_method','-rm', type=str, default='NIMBUSR')
parser.add_argument('--nimbusr_model_type','-nmt', type=str, default='offsets')
parser.add_argument('--output_folder','-o', type=str, default='results_J-MTPD')
parser.add_argument('--architecture','-a', type=str, default='two_branches')
parser.add_argument('--save_video', action='store_true', help='whether to save the video or not', default=False)
parser.add_argument('--focal_length', '-f', type=float, help='given focal length', default=0)
parser.add_argument('--gamma_factor', '-gf', type=float, help='gamma factor', default=1.0)

args = parser.parse_args()

GPU = 0

def load_nimbusr_net(type='offsets'):
    opt_net = { "n_iter": 8
        , "h_nc": 64
        , "in_nc": 4 #2 if args.gray else 4 #4
        , "out_nc":3 #1 if args.gray else 3 #3
        #, "ksize": 25
        , "nc": [64, 128, 256, 512]
        , "nb": 2
        , "gc": 32
        , "ng": 2
        , "reduction" : 16
        , "act_mode": "R" 
        , "upsample_mode": "convtranspose" 
        , "downsample_mode": "strideconv"}

    path_pretrained = args.restoration_network #r'../model_zoo/NIMBUSR.pth'
    
    if type=='pmbm':
        netG = net(n_iter=opt_net['n_iter'],
                    h_nc=opt_net['h_nc'],
                    in_nc=opt_net['in_nc'],
                    out_nc=opt_net['out_nc'],
                    nc=opt_net['nc'],
                    nb=opt_net['nb'],
                    act_mode=opt_net['act_mode'],
                    downsample_mode=opt_net['downsample_mode'],
                    upsample_mode=opt_net['upsample_mode']
                    )
    elif type=='offsets':
        netG = net_nimbusr_offsets(n_iter=opt_net['n_iter'],
            h_nc=opt_net['h_nc'],
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nc=opt_net['nc'],
            nb=opt_net['nb'],
            act_mode=opt_net['act_mode'],
            downsample_mode=opt_net['downsample_mode'],
            upsample_mode=opt_net['upsample_mode']
            )

    if os.path.exists(path_pretrained):
        print('Loading model for G [{:s}] ...'.format(path_pretrained))
        netG.load_state_dict(torch.load(path_pretrained))
    else:
        print('Model does not exists')
        
    netG = netG.to('cuda')

    return netG
    
def pad_input(inp_image, multiplicative_factor):
    B,C,M, N = inp_image.shape
    pad_rows = 0
    pad_cols = 0

    if (M % multiplicative_factor) == 0:
        new_M = M
    else:
        new_M = M - (M % multiplicative_factor) + multiplicative_factor
    if (N % multiplicative_factor) == 0:
        new_N = N
    else:
        new_N = N - (N % multiplicative_factor) + multiplicative_factor


    if (new_M - M) > 0 or (new_N - N) > 0:
        pad_rows = new_M - M
        pad_cols = new_N - N
        inp_image = F.pad(inp_image, (0, new_N - N, 0, new_M - M), 'reflect')
    
    return inp_image, pad_rows, pad_cols
    
if args.blurry_image.endswith('.txt'):
    with open(args.blurry_image) as f:
        blurry_images_list =  f.readlines()
        blurry_images_list = [file[:-1] for file in blurry_images_list]
        #blurry_images_list = blurry_images_list[48:]
else:
    blurry_images_list = [args.blurry_image]
    

if args.restoration_method=='NIMBUSR':
    netG = load_nimbusr_net(args.nimbusr_model_type)
    netG.eval()
    noise_level = 0.01
    noise_level = torch.FloatTensor([noise_level]).view(1,1,1).cuda(GPU)  
elif args.restoration_method=='RL':
    n_iters = 20   

reblur_model = args.reblur_model 
#sharp_image_filename = '/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1//sharp/000000000009_0.jpg'

n_positions = 25
restoration_method=args.restoration_method
output_folder=args.output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
with open(os.path.join(args.output_folder, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


offsets_net = OffsetsNet(n_offset=n_positions,offset_mode='raw').cuda(GPU)

state_dict = torch.load(args.reblur_model )
offsets_net.load_state_dict(state_dict)
offsets_net.eval()

for blurry_image_filename in blurry_images_list:    


    print(blurry_image_filename)
    img_name, ext = blurry_image_filename.split('/')[-1].split('.')   
        
    blurry_image = rescale(imread(blurry_image_filename)/255.0, (args.rescale_factor,args.rescale_factor,1),anti_aliasing=True)
    #sharp_image = rescale(imread(sharp_image_filename)/255.0,(0.6,0.6,1),anti_aliasing=True)
    blurry_tensor = torch.from_numpy(blurry_image).permute(2,0,1)[None].cuda(GPU).float()
    blurry_tensor, pad_rows, pad_cols = pad_input(blurry_tensor, 8)
    #sharp_tensor = torch.from_numpy(sharp_image).permute(2,0,1)[None].cuda(GPU).float()
    initial_tensor = blurry_tensor.clone()
    
    _, C,H,W = blurry_tensor.shape
    print(C,H,W)

    
    with torch.no_grad():
        blurry_tensor_ph = blurry_tensor**args.gamma_factor
        offsets = offsets_net(blurry_tensor_ph - 0.5)
        offsets = offsets[:,:,:H,:W]
        blurry_tensor = blurry_tensor[:,:,:H,:W]
	




    with torch.no_grad():
        output_ph = netG(blurry_tensor_ph[:,:,:H,:W], offsets[:,:,:H,:W], sf=1, sigma=noise_level[None,:])
        output = torch.clamp(output_ph,0,1)**(1.0/args.gamma_factor) 



        

    print(f'{img_name} range:', output.min(), output.max())
    output_img = tensor2im(torch.clamp(output[0,:,:H,:W].detach(),0,1) - 0.5)
    save_image(output_img, os.path.join(output_folder, img_name + '_PMBM.png' ))

    save_motion_flow(offsets.reshape(n_positions,2,H,W), os.path.join(output_folder, f'{img_name}_motion_flow.png'))
    #save_kernels_from_offsets(offsets, os.path.join(output_folder, f'{img_name}_kernels.png'))

    show_kernels_from_offsets_on_blurry_image(blurry_tensor[0],offsets[0].reshape(n_positions,2,H,W), os.path.join(output_folder, img_name + '_kernels.png' ))
    
    
    save_image((255*blurry_image).astype(np.uint8), os.path.join(output_folder, img_name + '.png' ))
    print('Output saved in ', os.path.join(output_folder, img_name + '_offsets.png' ))

 
    kh, kw = int(np.sqrt(n_positions)),int(np.sqrt(n_positions))   # we have 25 offsets
    weight = torch.ones((C,1,kh, kw)).cuda()/(kh*kw)
    #weight = torch.zeros((C,1,kh, kw)).cuda()
    #weight[:,0,4,4]=1
    #mask = torch.ones(1, kh * kw, H, W).cuda()
    
    #reblurred_image_ph = deform_conv2d(output_ph, offsets_dc[:,:,:H,:W], weight, padding=kh//2)
    reblurred_image_ph = reblur_offsets(output_ph, offsets.reshape(1,n_positions,2,H,W))
    reblurred_image = torch.clamp(reblurred_image_ph,0,1)**(1.0/args.gamma_factor)
    reblurred = tensor2im(torch.clamp(reblurred_image[0,:,:H,:W].detach(),0,1) - 0.5)
    save_image(reblurred, os.path.join(output_folder, img_name + '_reblurred.png' ))  
        
        
