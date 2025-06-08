import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from utils.emasc import EMASC

from ldm.data.cp_dataset import CPDataset
from ldm.resizer import Resizer
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.deepfashions import DFPairDataset
import torchgeometry as tgm
from torch import nn
from utils.data_utils import mask_features


import clip
from torchvision.transforms import Resize

from torch.nn import functional as F

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="which gpu to use",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=30,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--use_T_repaint",
        type=bool,
        default=True,
        help="use_T_repaint",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--unpaired",
        action='store_true',
        help="if enabled, uses the same starting code across samples "
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        help="path to dataroot of the dataset",
        default=""
    )
    parser.add_argument(
        "--ckpt_elbm_path",
        type=str,
        help="path to ckpt of elbm",
        default=""
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda:{}".format(opt.gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    config = OmegaConf.load(f"{opt.config}")
    version = opt.config.split('/')[-1].split('.')[0]
    model = load_model_from_config(config, f"{opt.ckpt}")
    # model = model.to(device)
    dataset = CPDataset(opt.dataroot, opt.H, mode='test', unpaired=opt.unpaired)
    loader = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False, num_workers=4, pin_memory=True)

    vae_normalize  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

    
    model.vae.decoder.blend_fusion = nn.ModuleList()
    feature_channels = [ 512, 512, 512, 256, 128]

    for blend_in_ch, blend_out_ch in zip(feature_channels, feature_channels):
        model.vae.decoder.blend_fusion.append(nn.Conv2d(blend_in_ch, blend_out_ch, kernel_size=3, bias=True, padding=1, stride=1))
    model.vae.use_blend_fusion = True
    model.vae.load_state_dict(torch.load(os.path.join(opt.ckpt_elbm_path,'checkpoint/checkpoint-40000/pytorch_model_1.bin')))

    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]
    int_layers = [1, 2, 3, 4, 5]

    emasc = EMASC(in_feature_channels,
                  out_feature_channels,
                  kernel_size=3,
                  padding=1,
                  stride=1,
                  type='nonlinear')
    emasac_sd = torch.load(os.path.join(opt.ckpt_elbm_path,'emasc_40000.pth'))
    emasc.load_state_dict(emasac_sd)

    emasc.cuda()
    emasc.eval()
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    result_path = os.path.join(outpath, "result")
    os.makedirs(result_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    
    # up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cuda()

    iterator = tqdm(loader, desc='Test Dataset', total=len(loader))
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data in iterator:
                    mask_tensor = data['inpaint_mask']
                    inpaint_image = data['inpaint_image']
                    ref_tensor = data['ref_imgs']
                    feat_tensor = data['warp_feat']
                    image_tensor = data['GT']
                    pose = data['pose']
                    sobel_img = data['sobel_img']
                    parse_agnostic = data['parse_agnostic']
                    warp_mask = data['warp_mask']
                    new_mask = warp_mask #+ mask_tensor
                    resize = transforms.Resize((opt.H, int(opt.H / 256 * 192)))

                    if opt.unpaired:
                        cm = data['cloth_mask']['unpaired'] 
                        c = data['cloth']['unpaired'] 
                    else:
                        cm = data['cloth_mask']['paired'] 
                        c = data['cloth']['paired'] 

                    test_model_kwargs = {}
                    test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
                    test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
                    test_model_kwargs['warp_feat'] = feat_tensor.to(device)
                    test_model_kwargs['new_mask'] = new_mask.to(device)
                    feat_tensor = feat_tensor.to(device)
                    ref_tensor = ref_tensor.to(device)

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector
                        uc = uc.repeat(ref_tensor.size(0), 1, 1)

                    c_vae = model.encode_first_stage(vae_normalize(ref_tensor.to(torch.float16)))
                    c_vae = model.get_first_stage_encoding(c_vae).detach()
                    c, patches = model.get_learned_conditioning(clip_normalize(ref_tensor.to(torch.float16)))
                    patches = model.fuse_adapter(patches,c_vae)
                    c = model.proj_out(c)
                    patches = model.proj_out_patches(patches)
                    c = torch.cat([c, patches], dim=1)
                 
                    down_block_additional_residuals = list()
                    mask_resduial = model.adapter_mask(parse_agnostic.cuda())
                    sobel_resduial = model.adapter_canny(sobel_img.cuda())
                    
                    for i in range(len(mask_resduial)):
                        down_block_additional_residuals.append(torch.cat([mask_resduial[i].unsqueeze(0), sobel_resduial[i].unsqueeze(0)],dim=0)) 

                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])
                    
                    warp_feat = model.encode_first_stage(feat_tensor)
                    warp_feat = model.get_first_stage_encoding(warp_feat).detach()

                    ts = torch.full((1,), 999, device=device, dtype=torch.long)
                    start_code = model.q_sample(warp_feat, ts)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     down_block_additional_residuals=down_block_additional_residuals,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code,
                                                     use_T_repaint = opt.use_T_repaint,
                                                     test_model_kwargs=test_model_kwargs)
                    samples_ddim = 1/ 0.18215 * samples_ddim
                    _, intermediate_features = model.vae.encode(data["im_mask"].cuda())
                    intermediate_features = [intermediate_features[i] for i in int_layers]
                    # Use EMASC
                    processed_intermediate_features = emasc(intermediate_features)
                    processed_intermediate_features = mask_features(processed_intermediate_features,(1- data["inpaint_mask"]).cuda())
                    processed_intermediate_features = processed_intermediate_features #* 0.18215
                    x_samples_ddim = model.vae.decode(samples_ddim, processed_intermediate_features, int_layers).sample
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    x_result = x_checked_image_torch

                    resize = transforms.Resize((opt.H, int(opt.H / 256 * 192)))

                    if not opt.skip_save:

                        def un_norm(x):
                            return (x + 1.0) / 2.0

                        for i, x_sample in enumerate(x_result):
                            filename = data['file_name'][i]
                            # filename = data['file_name']
                            save_x = resize(x_sample)
                            save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(save_x.astype(np.uint8))
                            img.save(os.path.join(result_path, filename[:-4] + ".png"))
                        
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()