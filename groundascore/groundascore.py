# %%
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union, Optional, List

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from PIL import Image
#from tqdm.notebook import tqdm
from IPython.display import display, clear_output
from tqdm import tqdm

from groundingdino.inference_on_a_image import main_run
import time
import os
from groundascore.utils import register_attention_control,create_directory_if_not_exists,log_image_optimization_params
from groundascore.utils import find_phrase_word_indices,find_difference
from groundascore.loss import CutLoss
import einops
from datetime import datetime, timedelta
import pytz

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]
device = torch.device('cuda:0')






def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]    
    h, w, c = image.shape
    left = min(left, w-1) #0
    right = min(right, w - left - 1)#w-1
    top = min(top, h - left - 1)#0
    bottom = min(bottom, h - top - 1)#h-1
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image




@torch.no_grad()
def get_text_embeddings(pipe: StableDiffusionPipeline, text: str) -> T:
    tokens = pipe.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                   return_tensors="pt", return_overflowing_tokens=True).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()

@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]

@torch.no_grad()
def decode(latent: T, pipeline: StableDiffusionPipeline, im_cat: TN = None):
    image = pipeline.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    return Image.fromarray(image)

def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:

    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for name, params in unet.named_parameters():
        if 'attn1' in name: # self-attention
            params.requires_grad = True
        else:
            params.requires_grad = False
    return unet, alphas, sigmas


class DDSLoss:
    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, alpha_t: T, sigma_t: T, get_raw=False,
                           guidance_scale=7.5):
        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])#null,null,source,target
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            e_t_no_cfg = e_t
            if get_raw:
                return e_t_uncond, e_t
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, (e_t_uncond,e_t_no_cfg)#pred_z0

    def get_sds_loss(self, z: T, text_embeddings: T, eps: TN = None, mask=None, t=None,
                 timestep: Optional[int] = None, guidance_scale=7.5) -> TS:
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, eps=eps, timestep=timestep)
            e_t, _ = self.get_eps_prediction(z_t, timestep, text_embeddings, alpha_t, sigma_t,
                                             guidance_scale=guidance_scale)
            grad_z = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (e_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            if mask is not None:
                grad_z = grad_z * mask
            log_loss = (grad_z ** 2).mean()
        sds_loss = grad_z.clone() * z
        del grad_z
        return sds_loss.sum() / (z.shape[2] * z.shape[3]), log_loss






    def get_dds_loss(self, z_source: T, z_target: T, text_emb_source: T, text_emb_target: T,bbox = None, mask =None,
                            eps=None, reduction='mean', symmetric: bool = False, calibration_grad=None, timestep: Optional[int] = None,
                      guidance_scale=7.5, raw_log=False,CDS_flag = False,reweight_flag = True) -> TS:
        with torch.no_grad():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(z_source, eps, timestep)
            z_t_target, _, _, _, _ = self.noise_input(z_target, eps, timestep)
            eps_pred, (eps_null, eps_pred_o) = self.get_eps_prediction(torch.cat((z_t_source, z_t_target)),
                                                    torch.cat((timestep, timestep)),
                                                    torch.cat((text_emb_source, text_emb_target)),
                                                    torch.cat((alpha_t, alpha_t)),
                                                    torch.cat((sigma_t, sigma_t)),
                                                    guidance_scale=guidance_scale,
                                                    )
            eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            eps_null_source, eps_null_target = eps_null.chunk(2)
            eps_pred_o_source, eps_pred_o_target = eps_pred_o.chunk(2)
            coeff = 5 #1./((eps_null_target - eps_pred_o_target).abs()).max()#5#min(1./((eps_null_target - eps_pred_o_target).abs()).max(),1./((eps_null_source - eps_pred_o_source).abs() * mask).max())
            if reweight_flag:
                gamma = ((((eps_null_target - eps_pred_o_target))*coeff).abs() * mask).mean().clamp(0,1)
            else:
                gamma = 1
            grad = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (eps_pred_target - eps_pred_source) * gamma
            if calibration_grad is not None:
                if calibration_grad.dim() == 4:
                    grad = grad - calibration_grad
                else:
                    grad = grad - calibration_grad[timestep - self.t_min]
            if raw_log:
                log_loss = eps.detach().cpu(), eps_pred_target.detach().cpu(), eps_pred_source.detach().cpu()
            else:
                log_loss = (grad ** 2).mean()

        # ================= BOX CDS BEGIN =============== #
        cutloss = 0
        if CDS_flag:
            with torch.enable_grad():
                sa_attn = dict()
                sa_attn[timestep.item()] = {}
                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention":
                        if "attn1" in name and "up" in name:
                            hidden_state = module.hs
                            sa_attn[timestep.item()][name] = hidden_state.detach().cpu()
                
                z_t_trg, _, _, _, _ = self.noise_input(z_target, eps, timestep)
                _ = self.unet(
                    z_t_trg,
                    timestep,
                    text_emb_target[0][1].unsqueeze(0)
                )
                
                cut_loss = CutLoss(256, [1, 2])
                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention":
                        # sa_cut
                        if "attn1" in name and "up" in name:
                            curr = module.hs
                            ref = sa_attn[timestep.item()][name].detach().to(device)
                            if curr.size(1) == 64**2:
                                bbox_scaled = bbox
                            elif curr.size(1) == 32**2:
                                bbox_scaled = [round(b/2) for b in bbox]
                            else:
                                bbox_scaled = [round(b/4) for b in bbox]
                            if bbox_scaled[1] == bbox_scaled[3]:
                                bbox_scaled[3] += 1
                            if bbox_scaled[0] == bbox_scaled[2]:
                                bbox_scaled[2] += 1                            
                            ref = einops.rearrange(ref, 'B (L L2) F -> B L L2 F', L=int(curr.size(1)**0.5))[:, bbox_scaled[1]:bbox_scaled[3], bbox_scaled[0]:bbox_scaled[2]]
                            curr = einops.rearrange(curr, 'B (L L2) F -> B L L2 F', L=int(curr.size(1)**0.5))[:, bbox_scaled[1]:bbox_scaled[3], bbox_scaled[0]:bbox_scaled[2]]
                            cutloss += cut_loss.get_attn_cut_loss(ref.float(), curr.float())
                # ================= BOX CDS DONE =============== #        

        loss = z_target * grad.clone()
        loss = loss * mask
        
        if symmetric:
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            loss_symm = self.rescale * z_source * (-grad.clone())
            loss += loss_symm.sum() / (z_target.shape[2] * z_target.shape[3])
        elif reduction == 'mean':
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        return loss, cutloss, log_loss, timestep

    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=torch.float32):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(device, dtype, pipe.unet, pipe.scheduler)
        self.prediction_type = pipe.scheduler.prediction_type



def calculate_intersection(box_a, box_b):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    if intersection_area == 0:
        return 0, 0, 0, 0, -1  # No intersection

    # Compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    box_a_ratio = intersection_area/box_a_area
    box_b_ratio = intersection_area/box_b_area
    max_index = np.argmax([box_a_ratio, box_b_ratio])
    return xA,yA,xB,yB,max_index




def image_optimization(pipeline: StableDiffusionPipeline, image: np.ndarray, texts_source: list,
                        texts_target: list, num_iters=200, bbox = None,output_dir ="logs/1111",reweight_flags = None,
                         beta = 1., cutloss_flag = None,edit_intensities =None) -> None:
    register_attention_control(pipeline)    
    dds_loss = DDSLoss(device, pipeline)
    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)
    bboxes_float = bbox

    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * 0.18215
        image_target = image_source.clone()
        embedding_null = get_text_embeddings(pipeline, "")
        embeddings_source = [get_text_embeddings(pipeline, text) for text in texts_source]
        embeddings_target = [get_text_embeddings(pipeline, text) for text in texts_target]
    z_source_dimensions = z_source.shape[2:4]
    bboxes_int = []
    masks = []
    for bbox in bboxes_float:
        x_min, y_min, x_max, y_max = bbox
        x_min_int = round(x_min * z_source_dimensions[1])  # Width
        y_min_int = round(y_min * z_source_dimensions[0])  # Height
        x_max_int = round(x_max * z_source_dimensions[1])  # Width
        y_max_int = round(y_max * z_source_dimensions[0])  # Height
        bboxes_int.append([x_min_int, y_min_int, x_max_int, y_max_int])
    
    for bbox, i in zip(bboxes_int, range(len(bboxes_int))):
        x_min_int, y_min_int, x_max_int, y_max_int = bbox
        mask = torch.zeros_like(z_source)
        if not i == len(bboxes_int) - 1:
            mask[:,:,y_min_int:y_max_int, x_min_int:x_max_int] = 1
            for other_bbox in bboxes_int:
                if other_bbox == bbox or other_bbox == [0, 0, 64, 64]:
                    continue
                xA,yA,xB,yB,max_index = calculate_intersection(bbox, other_bbox)
                if max_index == 1: #if other box ratio is higher
                    mask[:,:,yA:yB, xA:xB] = 0.3
        else:
            if i == 0:
                mask[:,:,:,:] = 1
            else:
                stacked_masks = torch.stack(masks)
                mask = torch.any(stacked_masks, dim=0)
        masks.append(mask)

    image_target.requires_grad = True
    z_taregt = z_source.clone()
    z_taregt.requires_grad = True
    optimizer = SGD(params=[z_taregt], lr=1e-1)

    for i in tqdm(range(num_iters)):
        for embedding_text, embedding_text_target,bbox_i, mask, j,CDS_flag, edit_intensity,reweight_flag in zip(embeddings_source, embeddings_target,
                                                         bboxes_int,masks,range(len(masks)),cutloss_flag,edit_intensities,reweight_flags): 
            if edit_intensity == 0:
                continue
            if beta == 0 and j == len(masks)-1:
                continue
            #total_loss = torch.tensor(0.).to(device)

            embedding_source = torch.stack([embedding_null, embedding_text], dim=1)
            embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

            loss, cutloss, _, timestep = dds_loss.get_dds_loss(z_source, z_taregt, embedding_source, embedding_target, bbox_i, mask, CDS_flag = CDS_flag,reweight_flag = reweight_flag)
            if isinstance(beta, list):
                time_section = torch.div(timestep, 200, rounding_mode='floor')#int(timestep//200)
                beta_t = beta[time_section]
            else:
                beta_t = beta
            if j == len(masks) - 1:
                loss = beta_t * loss

            optimizer.zero_grad()  # Reset gradients from previous iterations

            # Compute and backpropagate the primary loss
            (2000 * loss).backward()  # Backpropagate the main loss

            # Compute gradients for the cutloss without zeroing out existing gradients
            if cutloss != 0:
                grads = torch.autograd.grad(cutloss, z_taregt)[0]  # Compute gradients for cutloss
                masked_grads = grads * mask  # Apply the mask to the gradients

                # Manually update the gradients for z_target
                if z_taregt.grad is not None:
                    z_taregt.grad += masked_grads  # Accumulate with existing gradients
                else:
                    z_taregt.grad = masked_grads  # Set new gradients if none exist

            # Update parameters based on total accumulated gradients
            optimizer.step()
        if (i + 1) % 100 == 0:
            out = decode(z_taregt, pipeline, im_cat=image)
            out.save(os.path.join(output_dir, f"result_{i+1}.jpg"),"JPEG")
            del out
    out = decode(z_taregt, pipeline, im_cat=image)
    out.save(os.path.join(output_dir, f"result_{i+1}.jpg"),"JPEG")
    del out
        

def main(source_sentence, target_sentence, image_path,output_dir = "output/1111",num_iters = 500,model_id = "runwayml/stable-diffusion-v1-5",
         masked_DDS = True, beta = [0.5,0.4,0.3,0.2,0.1],grounding_sentences = None,
        bbox = None, cutloss_flag = None, edit_intensities = None,reweight_flags=None):
    image = load_512(image_path)

    timezone = pytz.timezone('Asia/Seoul')#type_your_timezone
    now = datetime.now(timezone)
    time_f_name = now.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir,time_f_name)
    create_directory_if_not_exists(output_dir)

    if masked_DDS:#run masked DDS
        if (not isinstance(source_sentence, list)) and (not isinstance(target_sentence, list)):
            text_source, text_target = find_difference(source_sentence, target_sentence)
            text_source.append(source_sentence)
            text_target.append(target_sentence)
        elif(isinstance(source_sentence, list)) and (isinstance(target_sentence, list)):
            text_source = source_sentence
            text_target = target_sentence
        else:
            raise ValueError("source and target sentences should be both list or string")

        if bbox == None:#make bbox using grounding dino
            if grounding_sentences == None:
                grounding_sentences = text_source
                grounding_sentences[-1] = ' and '.join(grounding_sentences[:-1])
            token_spans = find_phrase_word_indices(grounding_sentences[-1], grounding_sentences[:-1])
            config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            checkpoint_path = "groundingdino/weights/groundingdino_swint_ogc.pth"
            bbox = main_run(config_file , checkpoint_path , image_path , grounding_sentences[-1] , output_dir , token_spans= token_spans).tolist()
            torch.cuda.empty_cache() 
            bbox.append([0,0,1,1])

    else:#run original DDS or CDS without mask
        bbox = [[0,0,1,1]]
        text_source = [source_sentence]
        text_target = [target_sentence]
        beta = 1
        reweight_flags = [False]
    
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    if reweight_flags == None:
        reweight_flags = []
        text_source = text_source.copy()
        text_target = text_target.copy() 
        for box,i in zip(bbox,range(len(bbox))):
            if (box[3]-box[1])*(box[2]-box[0]) < 0.15:
                reweight_flags.append(True)
            else:
                reweight_flags.append(False)
    print("text_source: ",text_source)
    print("text_target: ",text_target)
    if cutloss_flag == None:
        cutloss_flag = [False]*len(text_source)
    if edit_intensities == None:
        edit_intensities = [1]*len(text_source)
    log_image_optimization_params(output_dir, text_source, text_target, num_iters, bbox, beta, cutloss_flag,image_path,reweight_flags)
    image_optimization(pipeline, image, text_source , text_target, num_iters = num_iters, bbox=bbox,
                       output_dir = output_dir, beta = beta,
                       cutloss_flag = cutloss_flag,reweight_flags = reweight_flags,edit_intensities = edit_intensities)
