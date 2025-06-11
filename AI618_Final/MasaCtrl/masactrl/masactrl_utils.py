import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor



from typing import Optional, Union, Tuple, List, Callable, Dict

from torchvision.utils import save_image
from einops import rearrange, repeat


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'CrossAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count


# class HijackProcessor(AttnProcessor):
#     def __init__(self, editor):
#         super().__init__()
#         self.editor = editor

#     def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
#         is_cross = encoder_hidden_states is not None
#         context = encoder_hidden_states if is_cross else hidden_states

#         q = attn.to_q(hidden_states)
#         k = attn.to_k(context)
#         v = attn.to_v(context)

#         bsz, tgt_len, _ = q.shape
#         query = q.view(bsz, tgt_len, attn.heads, -1).transpose(1, 2)        # [B, H, T, D]
#         key = k.view(bsz, k.shape[1], attn.heads, -1).transpose(1, 2)
#         value = v.view(bsz, v.shape[1], attn.heads, -1).transpose(1, 2)

#         scale = attn.scale
#         sim = torch.matmul(query, key.transpose(-2, -1)) * scale

#         if attention_mask is not None:
#             attention_mask = attention_mask[:, None, None, :]
#             sim = sim + attention_mask

#         attn_probs = F.softmax(sim, dim=-1)

#         # Default output: normal attention
#         out = torch.matmul(attn_probs, value)

#         # Hijack — allow editor to override
#         edited = self.editor(
#             q=query.reshape(-1, query.shape[-2], query.shape[-1]),
#             k=key.reshape(-1, key.shape[-2], key.shape[-1]),
#             v=value.reshape(-1, value.shape[-2], value.shape[-1]),
#             sim=sim.reshape(-1, sim.shape[-2], sim.shape[-1]),
#             attn=attn_probs.reshape(-1, attn_probs.shape[-2], attn_probs.shape[-1]),
#             is_cross=is_cross,
#             place_in_unet="controlnet",
#             num_heads=attn.heads
#         )

#         # Use editor's output if it returns a tensor
#         if edited is not None:
#             out = edited.view(bsz, attn.heads, tgt_len, -1).transpose(1, 2).reshape(bsz, tgt_len, -1)
#         else:
#             out = out.transpose(1, 2).reshape(bsz, tgt_len, -1)

#         return attn.to_out(out)


def register_attention_editor_controlnet(controlnet, editor: AttentionBase):
    """
    Monkey-patch every Cross-/Self-Attention layer in a ControlNet model
    so that it calls the given `editor` (AttentionStore, MutualSelfAttentionControl …).

    The implementation mirrors `regiter_attention_editor_diffusers()` that
    already works for the UNet, but walks through
        • controlnet.down_blocks
        • controlnet.mid_block
        • controlnet.up_blocks   (present in some CN variants)

    It never touches ModuleList / Sequential containers, so the pipeline
    keeps working without `NotImplementedError` crashes.
    """
    # ---------- 1. build a wrapper around the original attention layer ----------
    def ca_forward(attn_module, place_in_unet):
        """
        Returns a patched forward() that:
          • reproduces the stock diffusers attention maths
          • delegates the result-building part to `editor(...)`
        """
        def forward(x, encoder_hidden_states=None, attention_mask=None, **kw):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
                is_cross = True
            else:
                context = x
                is_cross = False

            if attention_mask is not None:
                mask = attention_mask
            else:
                mask = None

            to_out = attn_module.to_out
            if isinstance(to_out, nn.ModuleList):
                to_out = to_out[0]

            h = attn_module.heads
            q = attn_module.to_q(x)
            k = attn_module.to_k(context)
            v = attn_module.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                          (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * attn_module.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg)

            attn = sim.softmax(dim=-1)

            # -------- call the external editor (store / hijack) ------------
            out = editor(
                q, k, v, sim, attn,
                is_cross=is_cross,
                place_in_unet=place_in_unet,
                num_heads=h,
                scale=attn_module.scale
            )

            return to_out(out)

        return forward

    # ---------- 2. recurse through sub-modules and patch attention ----------
    def patch_block(block, place_in_unet):
        for name, sub in block.named_children():
            cls_name = sub.__class__.__name__
            if cls_name in ("CrossAttention", "Attention"):
                sub.forward = ca_forward(sub, place_in_unet)
                patch_block.hooked += 1
            else:
                patch_block(sub, place_in_unet)  # recurse

    patch_block.hooked = 0

    # walk down / mid / up
    if hasattr(controlnet, "down_blocks"):
        for idx, blk in enumerate(controlnet.down_blocks):
            patch_block(blk, f"cn-down{idx}")
    if hasattr(controlnet, "mid_block"):
        patch_block(controlnet.mid_block, "cn-mid")
    if hasattr(controlnet, "up_blocks"):
        for idx, blk in enumerate(controlnet.up_blocks):
            patch_block(blk, f"cn-up{idx}")

    editor.num_att_layers = max(editor.num_att_layers, patch_block.hooked)
    print(f"[✅ ControlNet] attention editor registered to {patch_block.hooked} layers.")
