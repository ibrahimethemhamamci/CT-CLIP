from pathlib import Path
import copy
import math
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import grad as torch_grad
from torchvision import transforms as T, utils


import torchvision

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from transformer_maskgit.attention import Attention, Transformer, ContinuousPositionBias

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def cast_tuple(val, l = 1):
    return val if isinstance(val, tuple) else (val,) * l

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    device=torch.device('cuda')
    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def l2norm(t):
    return F.normalize(t, dim = -1)

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# ctvit - 3d ViT with factorized spatial and temporal attention made into an vqgan-vae autoencoder

def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    device=torch.device('cuda')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        discr_base_dim = 16,
        dim_head = 64,
        heads = 8,
        channels = 1,
        use_vgg_and_gan = True,
        vgg = None,
        discr_attn_res_layers = (16,),
        use_hinge_loss = True,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_height, p2 = patch_width)
        )

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
        )
        
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    def calculate_video_token_mask(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        first_frame_mask, rest_frame_mask = video_frame_mask[:, :1], video_frame_mask[:, 1:]
        rest_vq_mask = rearrange(rest_frame_mask, 'b (f p) -> b f p', p = self.temporal_patch_size)
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim = -1)), dim = -1)
        return repeat(video_mask, 'b f -> b (f hw)', hw = (h // ph) * (w // pw))

    def get_video_patch_shape(self, num_frames, include_first_frame = True):
        patch_frames = 0

        if include_first_frame:
            num_frames -= 1
            patch_frames += 1

        patch_frames += (num_frames // self.temporal_patch_size)

        return (patch_frames, *self.patch_height_width)

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens % tokens_per_frame) == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert (num_tokens > 0)

        pseudo_frames = num_tokens // tokens_per_frames
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    def copy_for_eval(self):
        device = next(self.parameters()).device
        device=torch.device('cuda')
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    #@remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    #@remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]
        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        device=torch.device('cuda')
        attn_bias = self.spatial_rel_pos_bias(h, w, device = device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        return tokens

    def decode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])

        # decode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.dec_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        # decode - spatial

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        device=torch.device('cuda')
        attn_bias = self.spatial_rel_pos_bias(h, w, device = device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # to pixels

        #first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]

        #first_frame = self.to_pixels_first_frame(first_frame_token)

        #rest_frames = self.to_pixels(rest_frames_tokens)

        recon_video = self.to_pixels(tokens)

        #recon_video = torch.cat((first_frame, rest_frames), dim = 2)

        return recon_video

    def forward(
        self,
        video,
        mask = None,
        return_recons = False,
        return_recons_only = False,
        return_discr_loss = False,
        apply_grad_penalty = True,
        return_only_codebook_ids = False,
        return_encoded_tokens=False
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4
        #print(video.shape)

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        device=torch.device('cuda')
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches

        #first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        #rest_frames_tokens = self.to_patch_emb(rest_frames)
        #tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)
        tokens = self.to_patch_emb(video)
        # save height and width in

        shape = tokens.shape
        *_, h, w, _ = shape

        # encode - spatial

        tokens = self.encode(tokens)

        # quantize

        tokens, packed_fhw_shape = pack([tokens], 'b * d')

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)

        tokens, indices, commit_loss = self.vq(tokens, mask = vq_mask)

        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        if return_encoded_tokens:
            return tokens
            
        recon_video = self.decode(tokens)

        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        if return_recons_only:
            return returned_recon

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video, recon_video)

        # prepare a random frame index to be chosen for discriminator and perceptual loss

        pick_frame_logits = torch.randn(b, f)

        if exists(mask):
            mask_value = -torch.finfo(pick_frame_logits.dtype).max
            pick_frame_logits = pick_frame_logits.masked_fill(~mask, mask_value)

        frame_indices = pick_frame_logits.topk(1, dim = -1).indices

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            video = pick_video_frame(video, frame_indices)
            recon_video = pick_video_frame(recon_video, frame_indices)

            recon_video = recon_video.detach()
            video.requires_grad_()

            transform = T.Compose([T.Resize(256)])

            recon_video = transform(recon_video)
            video = transform(video)

            
            #print("TEST")
            #print(recon_video.shape)


            recon_video_discr_logits, video_discr_logits = map(self.discr, (recon_video, video))

            discr_loss = self.discr_loss(recon_video_discr_logits, video_discr_logits)

            if apply_grad_penalty:
                gp = gradient_penalty(video, video_discr_logits)
                loss = discr_loss + gp

            if return_recons:
                return loss, returned_recon

            return loss

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, returned_recon

            return recon_loss

        # perceptual loss

        input_vgg_input = pick_video_frame(video, frame_indices)
        recon_vgg_input = pick_video_frame(recon_video, frame_indices)
        transform = T.Compose([T.Resize(256)])
        input_vgg_input = transform(input_vgg_input)
        recon_vgg_input=transform(recon_vgg_input)
        
        # handle grayscale for vgg
        
        if video.shape[1] == 1:
            input_vgg_input2, recon_vgg_input2 = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (input_vgg_input, recon_vgg_input))
        transform = T.Compose([T.Resize(256)])
        input_vgg_input2 = transform(input_vgg_input2)
        recon_vgg_input2 = transform(recon_vgg_input2)



        input_vgg_feats = self.vgg(input_vgg_input2)
        recon_vgg_feats = self.vgg(recon_vgg_input2)

        perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(self.discr(recon_vgg_input))

        # calculate adaptive weight

        last_dec_layer = self.to_pixels[0].weight

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        if return_recons:
            return loss, returned_recon

        return loss
