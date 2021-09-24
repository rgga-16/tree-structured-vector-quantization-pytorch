
import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F

import utils, seeder

import numpy as np, math, argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Tree-Structured Vector Quantization for Fast Texture Synthesis')
    parser.add_argument('--in_path',type=str,help='Path to input texture image.')
    parser.add_argument('--out_path',type=str,help='Path to save output texture image.')
    parser.add_argument('--in_size',default=None,type=int,help='Size of input texture image to resize to, if needed.')
    parser.add_argument('--out_size',default=None,type=int,help='Size of output texture image. By default, it is in_size.')
    parser.add_argument('--n_levels',default=4,type=int,help='Number of levels in pyramid.')
    parser.add_argument('--n_size', default=10,type=int,help='Neighborhood size.')
    parser.add_argument('--parent_n_size', default=None,type=int,help='Neighborhood size of previous level. By default, it is half of n_size.')
    return parser.parse_args()


def get_h_and_w(size):
    if type(size) is tuple:
        h, w = size
    else: 
        h = w =  size
    
    return h,w

def build_gaussian_pyramid(image,n_levels):
    gaussian_blur = GaussianBlur(kernel_size=(5,5))
    pyramid = [image]
    im = image.clone().detach().unsqueeze(0)
    for l in range(0,n_levels-1):
        blurred_im = gaussian_blur(im)
        downsampled_im = F.interpolate(blurred_im,scale_factor=0.5)
        im = downsampled_im
        pyramid.insert(0,im.squeeze())
    return pyramid

def get_neighborhood(curr_row,curr_col,image,n_size,exclude_curr_pixel=True):
    im = image.clone().detach()
    _,im_h,im_w = im.shape

    n_h,n_w = get_h_and_w(n_size)

    half_h = n_h//2
    half_w = n_w//2

    top_idx = curr_row-half_h
    bot_idx = curr_row+half_h
    left_idx = curr_col-half_w
    right_idx= curr_col+half_w

    top_pad = bot_pad=left_pad=right_pad=0
    
    if top_idx < 0: 
        top_pad = abs(top_idx)
    if bot_idx >= im_h:
        bot_pad = abs(im_h-1-bot_idx)
    if left_idx < 0:
        left_pad = abs(left_idx)
    if right_idx >= im_w:
        right_pad = abs(im_w-1-right_idx)

    top_idx = np.clip(curr_row-half_h,0,im_h-1)
    bot_idx = np.clip(curr_row+half_h,0,im_h-1)
    left_idx = np.clip(curr_col-half_w,0,im_w-1)
    right_idx= np.clip(curr_col+half_w,0,im_w-1)

    if exclude_curr_pixel:
        im[:,curr_row,curr_col]=0
        
    neighborhood = im[:,top_idx:bot_idx+1,left_idx:right_idx+1]
    neighborhood = F.pad(neighborhood,(left_pad,right_pad,top_pad,bot_pad),mode='constant')
    return neighborhood

def get_neighborhood_pyramid(curr_row,curr_col,pyramid,level,n_size,n_parent_size,exclude_curr_pixel):
    N = get_neighborhood(curr_row,curr_col,pyramid[level],n_size,exclude_curr_pixel)
    if level > 0:
        parent_row = curr_row//2
        parent_col = curr_col//2
        parent_N = get_neighborhood(parent_row,parent_col,pyramid[level-1],n_parent_size,False)
        if parent_N.shape[1] < N.shape[1]:
            diff = abs(N.shape[1] - parent_N.shape[1])
            parent_N = F.pad(parent_N,(int(diff/2),int(diff/2),int(diff/2),int(diff/2)),mode='constant')
        N = torch.stack((N,parent_N))

    return N

def get_neighborhood_pyramids(pyramid,level,n_size,n_parent_size,exclude_curr_pixel):
    kD_pixels=[]
    neighborhood_pyrs=[]
    image = pyramid[level]
    _,h,w = image.shape 
    for r in range(h):
        for c in range(w):
            kD_pixels.append(pyramid[level][:,r,c])
            N = get_neighborhood_pyramid(r,c,pyramid,level,n_size,n_parent_size,exclude_curr_pixel)
            neighborhood_pyrs.append(N)
    return torch.stack(neighborhood_pyrs),torch.stack(kD_pixels)


def tvsq(in_path,out_path,n_size,n_levels,in_size=None,out_size=None,parent_size=None):
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_h,n_w = get_h_and_w(n_size)
    if parent_size is None:
        parent_size = (math.ceil(n_h/2), math.ceil(n_w/2))
    
    # Load input texture image
    I_a = utils.image_to_tensor(utils.load_image(in_path),resize=in_size,device=d)

    if out_size is None: 
        o_h,o_w = I_a.shape[1:]
    else:
        o_h,o_w = get_h_and_w(out_size)
    
    # Initialize output texture image
    I_s = torch.rand(3,o_h,o_w,device=d).detach()
    
    # Build Gaussian pyramids for both images
    G_a = build_gaussian_pyramid(I_a,n_levels=n_levels)
    G_s = build_gaussian_pyramid(I_s,n_levels=n_levels)

    for L in range(n_levels):
        neighborhoods_pyr,kD_pixels = get_neighborhood_pyramids(G_a,L,n_size,parent_size,False)
        _,o_h,o_w = G_s[L].shape 
        for o_r in range(o_h):
            print(f'Rows {o_r+1} of {o_h}')
            for o_c in range(o_w):
                N_o = get_neighborhood_pyramid(o_r,o_c,G_s,L,n_size,parent_size,exclude_curr_pixel=True).unsqueeze(0)
                dists = F.pairwise_distance(N_o,neighborhoods_pyr,p=2,keepdim=True).squeeze()
                if L==0:
                    dists = torch.sum(dists,dim=(-1,-2))
                else: 
                    dists = torch.sum(dists,dim=(-1,-2,-3))
               
                best_idx = torch.argmin(dists)
                best_match = kD_pixels[best_idx]
                G_s[L][:,o_r,o_c]=best_match

    final_output = G_s[-1]

    out_im = utils.tensor_to_image(final_output)
    out_im.save(out_path)
    return out_im


def main():
    args = parse_arguments()
    tvsq(args.in_path,args.out_path,
        n_size=args.n_size,n_levels=args.n_levels,
        in_size=args.in_size,out_size=args.out_size,
        parent_size=args.parent_n_size)
    return

if __name__ == "__main__":   
    main()
