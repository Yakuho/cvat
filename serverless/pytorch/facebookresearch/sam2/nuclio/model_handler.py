# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import re
import io
import os
import cv2
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

import warnings

from PIL import Image
from redis import Redis
from cvat_sdk.api_client import ApiClient
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames

REDIS_TMPL_KEY_OBJ_PTR     = "ObjectsPtr:{jobId}-{frame_idx}-{objectId}"
REDIS_TMPL_KEY_MEMORY_OUTS = "MemOutputs:{jobId}-{frame_idx}-{objectId}"
REDIS_TMPL_KEY_IMAGE_FEATS = "ImageFeats:{jobId}-{frame_idx}"
REDIS_EX_OBJ_PTR           = 300
REDIS_EX_MEMORY_OUTS       = 300
REDIS_EX_IMAGE_FEATS       = 600

THRESHOLD = float(os.environ.get("THRESHOLD", 0.6))
BATCHSIZE = int(os.environ.get("BATCHSIZE", 1))


def model_gpumem(model):
    """calculate model memory to GPU"""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1 << 20)  # MB


def smart_device(required_gpumem: float):
    """smart select device"""
    if torch.cuda.is_available():
        profile = "Detected Devices "
        print(profile, end='')

        devices_info = [] # [dev-free-memery, ...] Device Memory (MB)
        for i in range(torch.cuda.device_count()):
            dname = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / (1 << 20)
            using = torch.cuda.memory_allocated(i) / (1 << 20)
            print("%sCUDA: %d (%s, %d MB) already use %d MB (%s%%)" % (
                ' ' * (len(profile) if i else 0), i, dname, total, using, round(using / total * 100, 2)))
            if total - using > required_gpumem:
                devices_info.append(total - using)

        if len(devices_info) > 0:
            bestid = np.argmax(devices_info)
            print("Found %d suitable GPUs, select CUDA: %d for the best device" % (len(devices_info), int(bestid)))
            device = torch.device('cuda:%d' % bestid)
        else:
            print("Available GPU free memory is insufficient for the model required VRAM, using CPU instead")
            device = torch.device('cpu')
    else:
        print("Detected no CUDA, using CPU instead")
        device = torch.device('cpu')
    return device


def torch_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().to(torch.float32).numpy()
    elif isinstance(data, dict):
        return {k: torch_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(torch_to_numpy(v) for v in data)
    return data


def numpy_to_torch(data, device: torch.device):
    if isinstance(data, np.ndarray):
        dtype = torch.get_autocast_dtype(str(device))
        return torch.from_numpy(np.array(data)).to(device).to(dtype)
    elif isinstance(data, dict):
        return {k: numpy_to_torch(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(numpy_to_torch(v, device) for v in data)
    return data


def torch_loads(content, device: torch.device) -> dict:
    data = dict(np.load(io.BytesIO(content), allow_pickle=True))
    return numpy_to_torch(data, device)


def torch_dumps(data: dict) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **torch_to_numpy(data))
    content = buffer.getvalue()
    buffer.close()
    return content


class ModelHandler:
    @property
    def device(self):
        return self._device

    def __init__(self, ckpt: str, cfg: str, client: ApiClient, redis: Redis):
        self._redis, self._client = redis, client

        model_name = ckpt.rsplit('/', 1)[-1]
        print("Loading Model Checkpoint:", model_name)

        self._predictor = build_sam2_video_predictor(cfg, ckpt)

        gpumem = model_gpumem(self._predictor)
        print("Model '%s' Required GPU Memory: %2s MB" % (model_name, round(gpumem, 2)))
        self._device = smart_device(gpumem * 1.2)  # 1.2 for Redundancy

        if self._device.type == "cuda":
            # use bfloat16 for the entire project
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs
            #   refer: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self._predictor.to(self._device)
        torch.set_default_device(self._device)

        self._resize = torchvision.transforms.Resize(
            (self._predictor.image_size, self._predictor.image_size))
        self._transform = torchvision.transforms.Compose([
            self._resize,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                # see load_video_frames in the SAM2 source
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    @torch.no_grad()
    def _get_image_feature(self, jobId: int, frame_idx: int):
        cache_key = REDIS_TMPL_KEY_IMAGE_FEATS.format(jobId=jobId, frame_idx=frame_idx)
        image_feature_cache = self._redis.get(cache_key)
        if image_feature_cache:
            self._redis.expire(cache_key, REDIS_EX_IMAGE_FEATS)
            data = torch_loads(image_feature_cache, self._device)

            reshape = lambda tensor, size: tensor.permute(1, 2, 0).reshape(tensor.shape[1], tensor.shape[2], *size)
            extract = lambda keyword: [data[k] for k in sorted([k for k in data.keys() if re.match(keyword, k)])]

            (   # Extra data from cache
                feat_sizes, imgsz, vision_feats, vision_pos_embeds
            ) = data["feat_sizes"], data["imgsz"], extract("vision_feats_\d+"), extract("vision_pos_embeds_\d+")
            feat_sizes = feat_sizes.to(torch.int).cpu().tolist() if isinstance(feat_sizes, torch.Tensor) else feat_sizes
            imgsz = imgsz.to(torch.int).cpu().tolist() if isinstance(imgsz, torch.Tensor) else imgsz

            backbone_out = {
                "backbone_fpn": [reshape(t, feat_sizes[i]) for i, t in enumerate(vision_feats)],
                "vision_pos_enc": [reshape(t, feat_sizes[i]) for i, t in enumerate(vision_pos_embeds)],
                "vision_features": reshape(vision_feats[-1], feat_sizes[-1]),
            }
        else:
            # Download image from cvat api (got io.File object)
            buffer, response = self._client.jobs_api.retrieve_data(
                jobId, type="frame", number=frame_idx, quality="original")

            image = Image.open(buffer)
            imgsz = list(image.size[::-1])  # wh -> hw
            image_tensor = self._transform(image)

            # Inference
            backbone_out = self._predictor.forward_image(image_tensor.unsqueeze(0).to(self._device))
            _, vision_feats, vision_pos_embeds, feat_sizes = self._predictor._prepare_backbone_features(backbone_out)
            feat_sizes = [list(fs) for fs in feat_sizes]  # align type for get cache

            # Try to clean image temp file
            buffer.close()
            try:
                os.remove(buffer.name)
            except PermissionError:
                warnings.warn("PermissionError: fail to delete cvat temp image file %s." % buffer.name)

            # Cache result
            self._redis.set(cache_key, torch_dumps({
                **{"vision_feats_%d" % i: t for i, t in enumerate(vision_feats)},
                **{"vision_pos_embeds_%d" % i: t for i, t in enumerate(vision_pos_embeds)},
                "feat_sizes": feat_sizes,
                "imgsz": imgsz,
            }), ex=REDIS_EX_IMAGE_FEATS)

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes, imgsz

    @torch.no_grad()
    def _get_image_memoutput(self, jobId: int, frame_idx: int, annotation_idx: int, objectId: str):
        cache_key = REDIS_TMPL_KEY_MEMORY_OUTS.format(jobId=jobId, frame_idx=frame_idx, objectId=objectId)
        memory_out_cache = self._redis.get(cache_key)
        if memory_out_cache:
            self._redis.expire(cache_key, REDIS_EX_MEMORY_OUTS)
            data = torch_loads(memory_out_cache, self._device)
            maskmem_pos_enc = [data[k] for k in sorted([k for k in data.keys() if re.match("maskmem_pos_enc_", k)])]
            maskmem_features = data["maskmem_features"]
            low_res_mask = data["low_res_mask"]
        else:
            backbone_out, vision_feats, vision_pos_embeds, feat_sizes, imgsz = self._get_image_feature(jobId, frame_idx)

            # Loading annotation from cvat-client
            annotations, response = self._client.jobs_api.retrieve_annotations(jobId)
            shape = [s for s in annotations.shapes if s["id"] == annotation_idx].pop(0)

            # Build Mask
            mask = np.zeros(imgsz, dtype=np.uint8)
            if str(shape.type) == "mask":  # RLE
                x0, y0, x1, y1 = map(int, shape.points[-4:])  # shape.points = [rle] + [x0, y0, x1, y1]
                mask_part = [v for i, n in enumerate(shape.points[:-4]) for v in [int(i) % 2] * int(n)]
                mask_part = np.array(mask_part).reshape((y1 - y0 + 1, x1 - x0 + 1))
                mask[y0: y1 + 1, x0: x1 + 1] = mask_part
            elif str(shape.type) == "polygon":
                cv2.drawContours(mask, [np.array(shape["points"], np.int32).reshape(-1, 2)], -1, 1, cv2.FILLED)
            else:
                raise TypeError("SAM2.1 AutoTrack not supported type: %s" % shape["type"])
            mask = torch.Tensor(mask[None, None, ...])  # mask to tensor(B, C, H, W)

            # Simulate predict result
            low_res_mask = F.interpolate(
                mask, mode="bilinear", align_corners=False, size=(256, 256))
            high_res_mask = F.interpolate(
                mask, mode="bilinear", align_corners=False, size=(self._predictor.image_size, self._predictor.image_size))
            high_res_mask[high_res_mask > 0] = +18.0
            high_res_mask[high_res_mask < 0] = -18.0
            object_score_logits = torch.Tensor([[8.0]])

            # Inference memory encoder
            maskmem_features, maskmem_pos_enc = self._predictor._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_mask.to(self._device),
                object_score_logits=object_score_logits.to(self._device),
                is_mask_from_pts=True
            )

            # Cache result
            self._redis.set(cache_key, torch_dumps({
                **{"maskmem_pos_enc_%d" % i: t for i, t in enumerate(maskmem_pos_enc)},
                "maskmem_features": maskmem_features,
                "low_res_mask": low_res_mask,
            }), ex=REDIS_EX_MEMORY_OUTS)

        return dict(maskmem_features=maskmem_features, maskmem_pos_enc=maskmem_pos_enc, maskmem_pred_masks=low_res_mask)

    @torch.no_grad()
    def _prepare_memory_conditioned_features(self, jobId: int, frame: int, objectId: str, cond: dict, non_cond: dict):
        """reference: sam2/modeling/sam2_base.py function: SAM2Base._prepare_memory_conditioned_features"""
        frame_idx = frame
        num_frames = len(cond) + len(non_cond) + 1
        cond = {int(k): v for k, v in cond.items()}  # force str to int
        non_cond = {int(k): v for k, v in non_cond.items()}  # force str to int

        track_in_reverse = frame_idx < sorted([*cond.keys(), *non_cond.keys()])[0]
        backbone_out, vision_feats, vision_pos_embeds, feat_sizes, imgsz = self._get_image_feature(jobId, frame_idx)
        B, C, H, W = backbone_out["backbone_fpn"][-1].shape
        if self._predictor.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = backbone_out["backbone_fpn"][-1]
            return pix_feat

        num_obj_ptr_tokens = 0
        # Retrieve the memories encoded with the maskmem backbone
        to_cat_memory, to_cat_memory_pos_embed = [], []
        # Add conditioning frame's output first (all cond frames have t_pos=0 for
        # when getting temporal positional embedding below)
        assert len(cond) > 0, "SAM2.1 requires at least one condition to track by memory."
        # Select a maximum number of temporally closest cond frames for cross attention
        selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
            frame_idx, cond, self._predictor.max_cond_frames_in_attn)
        t_pos_and_prevs = [(0, {"frame": i, **out}) for i, out in selected_cond_outputs.items()]
        stride = 1 if self._predictor.training else self._predictor.memory_temporal_stride_for_eval
        for t_pos in range(1, self._predictor.num_maskmem):
            t_rel = self._predictor.num_maskmem - t_pos  # how many frames before current frame
            if t_rel == 1:
                # for t_rel == 1, we take the last frame (regardless of r)
                if not track_in_reverse:
                    # the frame immediately before this frame (i.e. frame_idx - 1)
                    prev_frame_idx = frame_idx - t_rel
                else:
                    # the frame immediately after this frame (i.e. frame_idx + 1)
                    prev_frame_idx = frame_idx + t_rel
            else:
                # for t_rel >= 2, we take the memory frame from every r-th frames
                if not track_in_reverse:
                    # first find the nearest frame among every r-th frames before this frame
                    # for r=1, this would be (frame_idx - 2)
                    prev_frame_idx = ((frame_idx - 2) // stride) * stride
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                else:
                    # first find the nearest frame among every r-th frames after this frame
                    # for r=1, this would be (frame_idx + 2)
                    prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
            out = non_cond.get(prev_frame_idx, None)
            if out is None:
                # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                # frames, we still attend to it as if it's a non-conditioning frame.
                out = unselected_cond_outputs.get(prev_frame_idx, None)
            t_pos_and_prevs.append((t_pos, {"frame": prev_frame_idx, **out} if out else out))

        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue  # skip padding frames
            prev = self._get_image_memoutput(jobId, prev["frame"], prev["id"], objectId)
            feats = prev["maskmem_features"].to(self._device, non_blocking=True)
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
            # Temporal positional encoding
            maskmem_enc = prev["maskmem_pos_enc"][-1].to(self._device)
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
            maskmem_enc = maskmem_enc + self._predictor.maskmem_tpos_enc[
                self._predictor.num_maskmem - t_pos - 1]
            to_cat_memory_pos_embed.append(maskmem_enc)

        # Construct the list of past object pointers
        if self._predictor.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = min(num_frames, self._predictor.max_obj_ptrs_in_encoder)
            # Because cond frame have no info for obj_ptr in cvat normally, create empty
            ptr_cond_outputs, pos_and_ptrs = {}, []
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or t >= num_frames:
                    break
                out = non_cond.get(t, unselected_cond_outputs.get(t, None))
                if out is not None:
                    cache_key = REDIS_TMPL_KEY_OBJ_PTR.format(jobId=jobId, frame_idx=t, objectId=objectId)
                    obj_ptr_cache = self._redis.get(cache_key)
                    if obj_ptr_cache:
                        self._redis.expire(cache_key, REDIS_EX_OBJ_PTR)
                        data = torch_loads(obj_ptr_cache, self._device)
                        pos_and_ptrs.append((t_diff, data["obj_ptr"]))
            # If we have at least one object pointer, add them to the across attention
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                # a temporal positional embedding based on how far each object pointer is from
                # the current frame (sine embedding normalized by the max pointer num).
                if self._predictor.add_tpos_enc_to_obj_ptrs:
                    t_diff_max = max_obj_ptrs_in_encoder - 1
                    tpos_dim = C if self._predictor.proj_tpos_enc_in_obj_ptrs else self._predictor.mem_dim
                    obj_pos = torch.tensor(pos_list).to(device=self._device, non_blocking=True)
                    obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                    obj_pos = self._predictor.obj_ptr_tpos_proj(obj_pos)
                    obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self._predictor.mem_dim)
                else:
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self._predictor.mem_dim)
                if self._predictor.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self._predictor.mem_dim, self._predictor.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self._predictor.mem_dim, dim=0)
                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self._predictor.memory_attention(
            curr=vision_feats[-1],
            curr_pos=vision_pos_embeds[-1],
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens
        )

        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return (pix_feat_with_mem, *backbone_out["backbone_fpn"][:2], vision_feats, feat_sizes, imgsz)

    @torch.no_grad()
    def _batch_step(self, jobId: int, batch: list[dict]):
        # pix_feat_with_mem, batch_high_res_feature0, batch_high_res_feature1, vision_feats, feat_sizes, imgsz (h, w)
        results = [self._prepare_memory_conditioned_features(
            jobId, item["frame"], item["objectId"], item["cond"], item["non_cond"]) for item in batch]

        feat_sizes, batch_imgsz = [[t[i] for t in results] for i in (4, 5)]
        assert all(x == feat_sizes[0] for x in feat_sizes), "All feat_sizes must be identical in SAM2.1 Batch mode"
        vision_feats = [torch.cat(tensors=[t[3][i] for t in results], dim=1) for i in range(len(feat_sizes[0]))]
        batch_mem_pix_feat, batch_high_res_feature0, batch_high_res_feature1 = (
            torch.cat(tensors=[t[i] for t in results], dim=0) for i in (0, 1, 2)
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = self._predictor._forward_sam_heads(
            backbone_features=batch_mem_pix_feat,
            high_res_features=[batch_high_res_feature0, batch_high_res_feature1],
        )

        maskmem_features, maskmem_pos_enc = self._predictor._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes[0],  # take batch[0]
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False
        )

        for i, item in enumerate(batch):
            frame_idx, objectId = item["frame"], item["objectId"]

            # Save non-cond mask memory output
            cache_key = REDIS_TMPL_KEY_MEMORY_OUTS.format(jobId=jobId, frame_idx=frame_idx, objectId=objectId)
            self._redis.set(cache_key, torch_dumps({
                **{"maskmem_pos_enc_%d" % i: t[i: i + 1] for i, t in enumerate(maskmem_pos_enc)},
                "maskmem_features": maskmem_features[i: i + 1],
                "low_res_mask": low_res_masks[i: i + 1],
            }), ex=REDIS_EX_MEMORY_OUTS)

            # Save non-cond obj_ptr
            cache_key = REDIS_TMPL_KEY_OBJ_PTR.format(jobId=jobId, frame_idx=frame_idx, objectId=objectId)
            self._redis.set(cache_key, torch_dumps({
                "obj_ptr": obj_ptr[i: i + 1]
            }), ex=REDIS_EX_OBJ_PTR)

        return ious.squeeze(-1), low_res_masks, batch_imgsz

    def handle(self, jobId: int, batch: list[dict], threshold=THRESHOLD, batch_size=BATCHSIZE):
        confs, masks, img_sizes = self._batch_step(jobId, batch)
        for conf, mask, img_size, info in zip(confs, masks, img_sizes, batch):
            if conf < threshold:
                yield None
                continue

            mask = F.interpolate(mask[None, ...], mode="bilinear", align_corners=False, size=img_size)
            mask = mask.cpu().numpy()[0, 0]  # drop (batch, channel) dims
            mask = (mask > 0).astype(np.uint8) * 255
            if mask[mask > 0].size <= 4:
                yield None
                continue

            if info["type"] == "polygon":
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                points = contours[int(np.argmax([cv2.contourArea(c) for c in contours]))].reshape(-1).tolist()
            elif info["type"] == "mask":
                coords = np.where(mask > 0)
                y0, x0 = coords[0].min(), coords[1].min()
                y1, x1 = coords[0].max(), coords[1].max()
                flat = mask[y0: y1, x0: x1].ravel()
                (run_indices,) = np.diff(flat, prepend=[not flat[0]], append=[not flat[-1]]).nonzero()
                run_lengths = np.diff(run_indices, prepend=[0]) if flat[0] else np.diff(run_indices)
                points = run_lengths.tolist() + list(map(int, [x0, y0, x1 - 1, y1 - 1]))
            else:
                raise NotImplementedError("type %s not support SAM2.1 AutoTrack." % info["type"])

            yield {"confidence": float(conf), "labelId": info["labelId"], "points": points, "type": info["type"]}
