import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.spatial_color_alignment as sca_utils
from utils.spatial_color_alignment import get_gaussian_kernel, match_colors
import lpips
from utils.ssim import cal_ssim
from utils.data_format_utils import numpy_to_torch, torch_to_numpy
import numpy as np
import lpips
from utils.warp import warp



class L2(nn.Module):
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_m = pred
        gt_m = gt

        if valid is None:
            mse = F.mse_loss(pred_m, gt_m)
        else:
            mse = F.mse_loss(pred_m, gt_m, reduction='none')

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse + 1e-6


class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = L2(boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, valid=None):
        assert pred.dim() == 4 and pred.shape == gt.shape
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr


#################################################################################
# Compute aligned L1 loss
#################################################################################

class AlignedL1(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        l1 = F.l1_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        l1 = l1 + eps
        elem_ratio = l1.numel() / valid.numel()
        l1 = (l1 * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return l1


class AlignedL1_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l1 = AlignedL1(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        
    def forward(self, pred, gt, burst_input):
        L1_all = [self.l1(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        L1_loss = sum(L1_all) / len(L1_all)
        return L1_loss


def make_patches(output, labels, burst, patch_size=48):    

    stride = patch_size-(burst.size(-1)%patch_size)
    
    burst1 = burst[0].unfold(2,patch_size,stride).unfold(3,patch_size,stride).contiguous()
    burst1 = burst1.view(14,4,burst1.size(2)*burst1.size(3),patch_size,patch_size).permute(2,0,1,3,4)            
 
    output1 = output.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    output1 = output1.view(3,output1.size(2)*output1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)

    labels1 = labels.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    labels1 = labels1[0].view(3,labels1.size(2)*labels1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)
    
    return output1, labels1, burst1

#################################################################################
# Compute aligned PSNR, LPIPS, and SSIM
#################################################################################


class AlignedPred(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = sca_utils.match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        return pred_warped_m, gt, valid


class AlignedL2(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = sca_utils.match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        mse = F.mse_loss(pred_warped_m, gt, reduction='none')
        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse

class AlignedL2_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        
    def forward(self, pred, gt, burst_input):
        L2_all = [self.l2(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        L2_loss = sum(L2_all) / len(L2_all)
        return L2_loss
        
class AlignedSSIM_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        

    def ssim(self, pred, gt, burst_input):
        
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)

        gt = gt[0, 0, :, :]
        pred_warped_m = pred_warped_m[0, 0, :, :]

        mssim,ssim_map = cal_ssim(pred_warped_m*255, gt*255)
        ssim_map = torch.from_numpy(ssim_map).float()
        valid = torch.squeeze(valid)

        eps = 1e-12
        elem_ratio = ssim_map.numel() / valid.numel()
        ssim = (ssim_map * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return 1 - ssim

    def forward(self, pred, gt, burst_input):
        ssim_all = [self.ssim(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        ssim = sum(ssim_all) / len(ssim_all)
        return ssim

class AlignedLPIPS_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    def lpips(self, pred, gt, burst_input):

        #### PSNR
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)
        var1 = 2*pred_warped_m-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_vgg(var1, var2)
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt, burst_input):
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips

class AlignedPSNR(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value    

    def psnr(self, pred, gt, burst_input):
        
        #### PSNR
        mse = self.l2(pred, gt, burst_input) + 1e-12
        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, burst_input):

        pred, gt, burst_input = make_patches(pred, gt, burst_input)
        psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr


class AlignedSSIM(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        

    def ssim(self, pred, gt, burst_input):
        
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)

        gt = gt[0, 0, :, :].cpu().numpy()
        pred_warped_m = pred_warped_m[0, 0, :, :].cpu().numpy()

        mssim,ssim_map = cal_ssim(pred_warped_m*255, gt*255)
        ssim_map = torch.from_numpy(ssim_map).float()
        valid = torch.squeeze(valid.cpu())

        eps = 1e-12
        elem_ratio = ssim_map.numel() / valid.numel()
        ssim = (ssim_map * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return ssim

    def forward(self, pred, gt, burst_input):

        pred, gt, burst_input = make_patches(pred, gt, burst_input)
        ssim_all = [self.ssim(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        ssim = sum(ssim_all) / len(ssim_all)
        return ssim

class AlignedLPIPS(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.loss_fn_alex = lpips.LPIPS(net='alex')

    def lpips(self, pred, gt, burst_input):

        #### PSNR
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)
        var1 = 2*pred_warped_m-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_alex(var1.cpu(), var2.cpu())
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt, burst_input):

        pred, gt, burst_input = make_patches(pred, gt, burst_input)
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips






