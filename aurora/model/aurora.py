"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
import warnings
from datetime import timedelta
from functools import partial
from typing import Optional
import torch.nn.functional as F

import torch
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from einops import rearrange
from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.lora import LoRAMode
from aurora.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone

__all__ = ["Aurora", "AuroraSmall", "AuroraHighRes"]


class Aurora(torch.nn.Module):
    """The Aurora model.

    Defaults to to the 1.3 B parameter configuration.
    """

    def __init__(
        self,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        out_surf_vars: tuple[str, ...] = None,
        out_atmos_vars: tuple[str, ...] = None,
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        max_history_size: int = 2,
        timestep: timedelta = timedelta(hours=6),
        stabilise_level_agg: bool = False,
        use_ours: bool = False,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        autocast: bool = False,
        task: str = "pre",
        ours_prompt_length=30
    ) -> None:
        """Construct an instance of the model.

        Args:
            surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model.
            static_vars (tuple[str, ...], optional): All static variables supported by the
                model.
            atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model.
            window_size (tuple[int, int, int], optional): Vertical height, height, and width of the
                window of the underlying Swin transformer.
            encoder_depths (tuple[int, ...], optional): Number of blocks in each encoder layer.
            encoder_num_heads (tuple[int, ...], optional): Number of attention heads in each encoder
                layer. The dimensionality doubles after every layer. To keep the dimensionality of
                every head constant, you want to double the number of heads after every layer. The
                dimensionality of attention head of the first layer is determined by `embed_dim`
                divided by the value here. For all cases except one, this is equal to `64`.
            decoder_depths (tuple[int, ...], optional): Number of blocks in each decoder layer.
                Generally, you want this to be the reversal of `encoder_depths`.
            decoder_num_heads (tuple[int, ...], optional): Number of attention heads in each decoder
                layer. Generally, you want this to be the reversal of `encoder_num_heads`.
            latent_levels (int, optional): Number of latent pressure levels.
            patch_size (int, optional): Patch size.
            embed_dim (int, optional): Patch embedding dimension.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks. The dimensionality of these attention heads will be equal to
                `embed_dim` divided by this value.
            mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs.
            drop_rate (float, optional): Drop-out rate.
            drop_path (float, optional): Drop-path rate.
            enc_depth (int, optional): Number of Perceiver blocks in the encoder.
            dec_depth (int, optioanl): Number of Perceiver blocks in the decoder.
            dec_mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs in the
                decoder. The embedding dimensionality here is different, which is why this is a
                separate parameter.
            perceiver_ln_eps (float, optional): Epsilon in the perceiver layer norm. layers. Used
                to stabilise the model.
            max_history_size (int, optional): Maximum number of history steps. You can load
                checkpoints with a smaller `max_history_size`, but you cannot load checkpoints
                with a larger `max_history_size`.
            timestep (timedelta, optional): Timestep of the model. Defaults to 6 hours.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
            lora_steps (int, optional): Use different LoRA adaptation for the first so-many roll-out
                steps.
            lora_mode (str, optional): LoRA mode. `"single"` uses the same LoRA for all roll-out
                steps, and `"all"` uses a different LoRA for every roll-out step. Defaults to
                `"single"`.
            surf_stats (dict[str, tuple[float, float]], optional): For these surface-level
                variables, adjust the normalisation to the given tuple consisting of a new location
                and scale.
            autocast (bool, optional): Use `torch.autocast` to reduce memory usage. Defaults to
                `False`.
        """
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or dict()
        self.autocast = autocast
        self.max_history_size = max_history_size
        
        self.timestep = timestep
        self.encoder_depths = encoder_depths
        self.decoder_depths = decoder_depths
        self.task = task
        if task =="downscale":
            out_surf_vars = surf_vars
            out_atmos_vars = atmos_vars
        
        if self.surf_stats:
            warnings.warn(
                f"The normalisation statics for the following surface-level variables are manually "
                f"adjusted: {', '.join(sorted(self.surf_stats.keys()))}. "
                f"Please ensure that this is right!",
                stacklevel=2,
            )

        self.encoder = Perceiver3DEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=embed_dim // num_heads,
            depth=enc_depth,
            latent_levels=latent_levels,
            max_history_size=max_history_size,
            perceiver_ln_eps=perceiver_ln_eps,
            stabilise_level_agg=stabilise_level_agg,
        )

        self.use_ours = use_ours
        if self.use_ours:
            embedding_aware_surf = torch.stack(tuple(self.encoder.surf_token_embeds.weights.values()), dim=1)
            embedding_aware_atmos = torch.stack(tuple(self.encoder.atmos_token_embeds.weights.values()), dim=1)
            embedding_aware = torch.cat((embedding_aware_surf, embedding_aware_atmos), dim=1).squeeze()
            aggregate_aware_q = self.encoder.level_agg.layers[0][0].to_q.weight
            aggregate_aware_kv = self.encoder.level_agg.layers[0][0].to_kv.weight
            aggregate_aware = F.scaled_dot_product_attention(aggregate_aware_q, aggregate_aware_kv, aggregate_aware_kv)
        else:
            embedding_aware = None
            aggregate_aware = None

        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
            use_ours=use_ours,
            lora_steps=lora_steps,
            lora_mode=lora_mode,
            embedding_aware=embedding_aware,
            aggregate_aware=aggregate_aware,
            ours_prompt_length=ours_prompt_length,
            
        )
        self.decoder = Perceiver3DDecoder(
            out_surf_vars=out_surf_vars,
            out_atmos_vars=out_atmos_vars,
            patch_size=patch_size,
            # Concatenation at the backbone end doubles the dim.
            embed_dim=embed_dim * 2,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            depth=dec_depth,
            # Because of the concatenation, high ratios are expensive.
            # We use a lower ratio here to keep the memory in check.
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
        )     

    def forward(self, batch: Batch = None) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        x = self.encoder(
            batch,
            lead_time=self.timestep,
        )

        if self.use_ours:
            embedding_aware_surf = torch.stack(tuple(self.encoder.surf_token_embeds.weights.values()), dim=1)
            embedding_aware_atmos = torch.stack(tuple(self.encoder.atmos_token_embeds.weights.values()), dim=1)
            embedding_aware = torch.cat((embedding_aware_surf, embedding_aware_atmos), dim=1).squeeze()
            aggregate_aware_q = self.encoder.level_agg.layers[0][0].to_q.weight
            aggregate_aware_kv = self.encoder.level_agg.layers[0][0].to_kv.weight
            aggregate_aware = F.scaled_dot_product_attention(aggregate_aware_q, aggregate_aware_kv, aggregate_aware_kv)
        else:
            embedding_aware = None
            aggregate_aware = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():
            x = self.backbone(
                x,
                lead_time=self.timestep,
                patch_res=patch_res,
                rollout_step=
                batch.metadata.rollout_step,
                embedding_aware=embedding_aware,
                aggregate_aware=aggregate_aware,
        )
        surf_preds, atmos_preds = self.decoder(
                                x,
                                batch,
                                lead_time=self.timestep,
                                patch_res=patch_res,
                            )
        return surf_preds, atmos_preds


    def load_checkpoint(self, repo: str, name: str, strict: bool = True) -> None:
        """Load a checkpoint from HuggingFace.

        Args:
            repo (str): Name of the repository of the form `user/repo`.
            name (str): Path to the checkpoint relative to the root of the repository, e.g.
                `checkpoint.cpkt`.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        path = hf_hub_download(repo_id=repo, filename=name)
        self.load_checkpoint_local(path, strict=strict)

    def load_checkpoint_local(self, path: str, strict: bool = True) -> None:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        # Assume that all parameters are either on the CPU or on the GPU.
        device = next(self.parameters()).device

        d = torch.load(path, map_location=device, weights_only=True)

        # You can safely ignore all cumbersome processing below. We modified the model after we
        # trained it. The code below manually adapts the checkpoints, so the checkpoints are
        # compatible with the new model.

        # Remove possibly prefix from the keys.
        for k, v in list(d.items()):
            if k.startswith("net."):
                del d[k]
                d[k[4:]] = v

        # Convert the ID-based parametrization to a name-based parametrization.
        if "encoder.surf_token_embeds.weight" in d:
            weight = d["encoder.surf_token_embeds.weight"]
            del d["encoder.surf_token_embeds.weight"]

            if weight.shape[-1]!=self.patch_size:
                print("interpolate surf encoder for changing patch size")
                weight = torch.nn.functional.interpolate(weight, size=weight.shape[2:3]+(self.patch_size,self.patch_size), mode='trilinear', align_corners=False)
            
            
            assert weight.shape[1] == 4 + 3
            for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
                if name == "2t" and "T2M" in self.surf_vars:
                    name = "T2M"
                    print(f"load 2t encoder weight for T2M")
                    
                elif name == "10u" and "U10M" in self.surf_vars:
                    name = "U10M"
                    print(f"load 10u encoder weight for U10M")
                    
                elif name == "10v" and "V10M" in self.surf_vars:
                    name = "V10M"
                    print(f"load 10v encoder weight for V10M")
                    
                elif name == "msl" and "MSL" in self.surf_vars:
                    nmae = "MSL"
                    print(f"load msl encoder weight for MSL")
                    
                d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]

        if "encoder.atmos_token_embeds.weight" in d:
            weight = d["encoder.atmos_token_embeds.weight"]
            del d["encoder.atmos_token_embeds.weight"]
            
            if weight.shape[-1]!=self.patch_size:
                print("interpolate atmos encoder for changing patch size")
                weight = torch.nn.functional.interpolate(weight, size=weight.shape[2:3]+(self.patch_size,self.patch_size), mode='trilinear', align_corners=False)

            assert weight.shape[1] == 5
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                if name == "z" and "Z" in self.atmos_vars:
                    name = "Z"
                    print("load z encoder weight for Z")
                    
                elif name == "u" and "U" in self.atmos_vars:
                    name = "U"
                    print("load u encoder weight for U")
                    
                elif name == "v" and "V" in self.atmos_vars:
                    name = "V"
                    print("load v encoder weight for V")
                    
                elif name == "t" and "T" in self.atmos_vars:
                    name = "T"
                    print("load t encoder weight for T")
                    
                elif name == "q" and "Q" in self.atmos_vars:
                    name = "Q"
                    print("load q encoder weight for Q")
                elif name == 'q' and 'r' in self.atmos_vars:
                    name = "r"
                    print("load q encoder weight for r")
                
                d[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]

        if "decoder.surf_head.weight" in d:
            weight = d["decoder.surf_head.weight"]
            bias = d["decoder.surf_head.bias"]
            del d["decoder.surf_head.weight"]
            del d["decoder.surf_head.bias"]

            if weight.shape[0] != 4 * self.patch_size**2:
                print("interpolate surf decoder for changing patch size")
                weight = weight.reshape(4, 4, 4, -1)
                weight = rearrange(weight, "p1 p2 v d -> v d p1 p2")
                weight = torch.nn.functional.interpolate(weight, size=(self.patch_size,self.patch_size), mode='bilinear', align_corners=False)
                weight = rearrange(weight, "v d p1 p2 -> p1 p2 v d").reshape(-1, 1024)
                
                bias = bias.reshape(4, 4, 4, -1)
                bias = rearrange(bias, "p1 p2 v d -> v d p1 p2")
                bias = torch.nn.functional.interpolate(bias, size=(self.patch_size,self.patch_size), mode='bilinear', align_corners=False)
                bias = rearrange(bias, "v d p1 p2 -> p1 p2 (v d)").reshape(-1)
                
            assert weight.shape[0] == 4 * self.patch_size**2
            assert bias.shape[0] == 4 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 4, -1)
            bias = bias.reshape(self.patch_size**2, 4)

            for i, name in enumerate(("2t", "10u", "10v", "msl")):
                if name == "2t" and "T2M" in self.surf_vars:
                    name = "T2M"
                    print(f"load 2t decoder weight for T2M")
                    
                elif name == "10u" and "U10M" in self.surf_vars:
                    name = "U10M"
                    print(f"load 10u decoder weight for U10M")
                    
                elif name == "10v" and "V10M" in self.surf_vars:
                    name = "V10M"
                    print(f"load 10v decoder weight for V10M")
                    
                elif name == "msl" and "MSL" in self.surf_vars:
                    nmae = "MSL"
                    print(f"load msl decoder weight for MSL")
                    
                d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]
                if self.task == "postprocess":
                    d[f"decoder.surf_heads.{name}_std.weight"] = weight[:, i]
                    d[f"decoder.surf_heads.{name}_std.bias"] = bias[:, i]

        if "decoder.atmos_head.weight" in d:
            weight = d["decoder.atmos_head.weight"]
            bias = d["decoder.atmos_head.bias"]
            del d["decoder.atmos_head.weight"]
            del d["decoder.atmos_head.bias"]

            if weight.shape[0] != 5 * self.patch_size**2:
                print("interpolate surf decoder for changing patch size")
                weight = weight.reshape(4, 4, 5, -1)
                weight = rearrange(weight, "p1 p2 v d -> v d p1 p2")
                weight = torch.nn.functional.interpolate(weight, size=(self.patch_size,self.patch_size), mode='bilinear', align_corners=False)
                weight = rearrange(weight, "v d p1 p2 -> p1 p2 v d").reshape(-1, 1024)
                
                bias = bias.reshape(4, 4, 5, -1)
                bias = rearrange(bias, "p1 p2 v d -> v d p1 p2")
                bias = torch.nn.functional.interpolate(bias, size=(self.patch_size,self.patch_size), mode='bilinear', align_corners=False)
                bias = rearrange(bias, "v d p1 p2 -> p1 p2 (v d)").reshape(-1)
                
            assert weight.shape[0] == 5 * self.patch_size**2
            assert bias.shape[0] == 5 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 5, -1)
            bias = bias.reshape(self.patch_size**2, 5)

            for i, name in enumerate(("z", "u", "v", "t", "q")):
                if name == "z" and "Z" in self.atmos_vars:
                    name = "Z"
                    print("load z decoder weight for Z")
                    
                elif name == "u" and "U" in self.atmos_vars:
                    name = "U"
                    print("load u decoder weight for U")
                    
                elif name == "v" and "V" in self.atmos_vars:
                    name = "V"
                    print("load v decoder weight for V")
                    
                elif name == "t" and "T" in self.atmos_vars:
                    name = "T"
                    print("load t decoder weight for T")
                    
                elif name == "q" and "Q" in self.atmos_vars:
                    name = "Q"
                    print("load q decoder weight for Q")
                elif name == "q" and "r" in self.atmos_vars:
                    name = "r"
                    print("load q decoder weight for r")
                    
                d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]
                
                if self.task == "postprocess":
                    d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
                    d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]
                
        if "encoder.atmos_latents" in d:
            weight = d["encoder.atmos_latents"]
            if self.encoder.atmos_latents.shape[0] != weight.shape[0]:
                print("laten")
            

        # Check if the history size is compatible and adjust weights if necessary.
        name = "2t"
        if "T2M" in self.surf_vars:
            name = "T2M"
        current_history_size = d[f"encoder.surf_token_embeds.weights.{name}"].shape[2]
        if self.max_history_size > current_history_size:
            self.adapt_checkpoint_max_history_size(d)
        elif self.max_history_size < current_history_size:
            raise AssertionError(
                f"Cannot load checkpoint with `max_history_size` {current_history_size} "
                f"into model with `max_history_size` {self.max_history_size}."
            )

        self.load_state_dict(d, strict=strict)

    def adapt_checkpoint_max_history_size(self, checkpoint: dict[str, torch.Tensor]) -> None:
        """Adapt a checkpoint with smaller `max_history_size` to a model with a larger
        `max_history_size` than the current model.

        If a checkpoint was trained with a larger `max_history_size` than the current model,
        this function will assert fail to prevent loading the checkpoint. This is to
        prevent loading a checkpoint which will likely cause the checkpoint to degrade is
        performance.

        This implementation copies weights from the checkpoint to the model and fills zeros
        for the new history width dimension. It mutates `checkpoint`.
        """
        print("adapt time")
        for name, weight in list(checkpoint.items()):
            # We only need to adapt the patch embedding in the encoder.
            enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
            enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
            if enc_surf_embedding or enc_atmos_embedding:
                # This shouldn't get called with current logic but leaving here for future proofing
                # and in cases where its called outside current context.
                if not (weight.shape[2] <= self.max_history_size):
                    raise AssertionError(
                        f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} "
                        f"into model with `max_history_size` {self.max_history_size}."
                    )
                D,_,T,p1,p2 = weight.shape
                
                new_weight = weight.repeat(1,1,self.max_history_size//T,1,1) / (self.max_history_size//T)

                checkpoint[name] = new_weight

    def configure_activation_checkpointing(self):
        """Configure activation checkpointing.

        This is required in order to compute gradients without running out of memory.
        """
        apply_activation_checkpointing(self, check_fn=lambda x: isinstance(x, BasicLayer3D))

    def get_num_layers(self):
        return sum(self.encoder_depths) + sum(self.decoder_depths)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

AuroraSmall = partial(
    Aurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
)

AuroraHighRes = partial(
    Aurora,
    patch_size=10,
    encoder_depths=(6, 8, 8),
    decoder_depths=(8, 8, 6),
)
