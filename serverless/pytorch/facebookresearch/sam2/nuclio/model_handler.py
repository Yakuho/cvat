# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import re
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
from cvat_sdk.api_client.model.labeled_shape import LabeledShape
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames
from utils import model_gpumem, smart_device, torch_loads, torch_dumps, Profile, group_frame
from enums import (CondFlattenItem, TrackState, TrackStateWithCondMemory,
                   ImageEmbeddingOutput, ImageFeature, MemoryOutput)

FEAT_SIZES = [[256, 256], [128, 128], [64, 64]]
REDIS_TMPL_KEY_OBJ_PTR     = "ObjectsPtr:{jobId}-{frame_idx}-{objectId}"
REDIS_TMPL_KEY_MEMORY_OUTS = "MemOutputs:{jobId}-{frame_idx}-{objectId}"
REDIS_TMPL_KEY_IMAGE_FEATS = "ImageFeats:{jobId}-{frame_idx}"
REDIS_EX_OBJ_PTR           = 300
REDIS_EX_MEMORY_OUTS       = 300
REDIS_EX_IMAGE_FEATS       = 600

# Reference:
#   Default Value: https://github.com/cvat-ai/cvat/blob/develop/cvat-ui/src/reducers/settings-reducer.ts#L41
#   Threshold Achieve: https://github.com/cvat-ai/cvat/blob/develop/cvat-core/src/opencv/opencv-interface.ts#L213
APPROX_THRESHOLD = float(os.environ.get("APPROX_THRESHOLD", 1.428571))  # (2.75*(13-9)-1)/7=1.428571
THRESHOLD = float(os.environ.get("THRESHOLD", 0.6))
BATCHSIZE = int(os.environ.get("BATCHSIZE", 1))


class SAMRedis(Redis):
    @staticmethod
    def _extract(data: dict, keyword: str):
        return [data[k] for k in sorted([k for k in data.keys() if re.match(keyword, k)])]

    def save_image_feature(self, keys: list[str],
                           data: tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]], list[tuple[int, int]]],
                           ex: int = REDIS_EX_IMAGE_FEATS):
        vision_feats, vision_pos_embeds, feat_sizes, imgsz = data

        pipeline = self.pipeline()
        for bn, key in enumerate(keys):
            pipeline.set(key, torch_dumps({
                **{"vision_feats_%d" % i     : t[:, bn: bn+1, :] for i, t in enumerate(vision_feats)},
                **{"vision_pos_embeds_%d" % i: t[:, bn: bn+1, :] for i, t in enumerate(vision_pos_embeds)},
                   "feat_sizes"              : feat_sizes,
                   "imgsz"                   : imgsz[bn],
            }), ex=ex)
        pipeline.execute()

    def save_object_memory(self, keys: list[str],
                           data: tuple[torch.Tensor, list[torch.Tensor], torch.Tensor],
                           ex: int = REDIS_EX_MEMORY_OUTS):
        maskmem_features, maskmem_pos_enc, low_res_mask = data

        pipeline = self.pipeline()
        for bn, key in enumerate(keys):
            pipeline.set(key, torch_dumps({
                **{"maskmem_pos_enc_%d" % i: t[bn: bn+1, ...] for i, t in enumerate(maskmem_pos_enc)},
                   "maskmem_features"      : maskmem_features[bn: bn + 1, ...],
                   "low_res_mask"          : low_res_mask[bn: bn + 1],
            }), ex=ex)
        pipeline.execute()

    def save_image_obj_ptr(self, keys: list[str],
                           data: torch.Tensor,
                           ex: int = REDIS_EX_OBJ_PTR):
        pipeline = self.pipeline()
        for bn, key in enumerate(keys):
            pipeline.set(key, torch_dumps(data[bn: bn + 1]), ex=ex)
        pipeline.execute()

    def load_image_feature(self, keys: list[str], device: torch.device, ex: int = REDIS_EX_IMAGE_FEATS) -> list[ImageFeature | None]:
        image_feature_cache = self.mget(keys)
        if not any(image_feature_cache):
            return [None] * len(image_feature_cache)

        results = []
        reshape = lambda tensor, size: tensor.permute(1, 2, 0).reshape(tensor.shape[1], tensor.shape[2], *size)

        for key, cache in zip(keys, image_feature_cache):
            if cache is None:
                results.append(cache)
                continue

            if isinstance(ex, int) and ex > 0:
                self.expire(key, ex)

            data = torch_loads(cache, device)

            imgsz             = data["imgsz"]
            feat_sizes        = [tuple(size) for size in data["feat_sizes"]]
            vision_feats      = self._extract(data, r"vision_feats_\d+")
            vision_pos_embeds = self._extract(data, r"vision_pos_embeds_\d+")

            feat_sizes = feat_sizes.to(torch.int).cpu().tolist() if isinstance(feat_sizes, torch.Tensor) else feat_sizes
            imgsz = imgsz.to(torch.int).cpu().tolist() if isinstance(imgsz, torch.Tensor) else imgsz

            backbone_out = ImageEmbeddingOutput.model_validate({
                "backbone_fpn"   : [reshape(t, feat_sizes[i]) for i, t in enumerate(vision_feats)],
                "vision_pos_enc" : [reshape(t, feat_sizes[i]) for i, t in enumerate(vision_pos_embeds)],
                "vision_features":  reshape(vision_feats[-1], feat_sizes[-1]),
            })

            results.append(ImageFeature(
                backbone_out=backbone_out,
                vision_feats=vision_feats,
                vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                imgsz=imgsz
            ))

        return results

    def load_object_memory(self, keys: list[str], device: torch.device, ex: int = REDIS_EX_IMAGE_FEATS) -> list[MemoryOutput | None]:
        memory_output_cache = self.mget(keys)
        if not any(memory_output_cache):
            return [None] * len(memory_output_cache)

        results = []

        for key, cache in zip(keys, memory_output_cache):
            if cache is None:
                results.append(cache)
                continue

            if isinstance(ex, int) and ex > 0:
                self.expire(key, ex)

            data = torch_loads(cache, device)

            maskmem_pos_enc  = self._extract(data, r"maskmem_pos_enc_\d+")
            maskmem_features = data["maskmem_features"]
            low_res_mask = data["low_res_mask"]

            memory_output = MemoryOutput.model_validate({
                "maskmem_features"  : maskmem_features,
                "maskmem_pos_enc"   : maskmem_pos_enc,
                "maskmem_pred_masks": low_res_mask
            })

            results.append(memory_output)

        return results

    def load_image_obj_ptr(self, keys: list[str], device: torch.device, ex: int = REDIS_EX_IMAGE_FEATS) -> list[torch.Tensor | None]:
        obj_ptr_cache = self.mget(keys)
        if not any(obj_ptr_cache):
            return [None] * len(obj_ptr_cache)

        results = []

        for key, cache in zip(keys, obj_ptr_cache):
            if cache is None:
                results.append(cache)
                continue

            if isinstance(ex, int) and ex > 0:
                self.expire(key, ex)

            data = torch_loads(cache, device)

            results.append(data)

        return results


class ModelHandler:
    def __init__(self, ckpt: str, cfg: str, client: ApiClient, redis: SAMRedis):
        self._redis, self._client = redis, client

        model_name = ckpt.rsplit('/', 1)[-1]
        print("Loading Model Checkpoint:", model_name)

        self._predictor = build_sam2_video_predictor(cfg, ckpt)

        gpumem = model_gpumem(self._predictor)
        print("Model '%s' Required GPU Memory: %2s MB" % (model_name, round(gpumem, 2)))
        self.device = smart_device(gpumem * 1.2)  # 1.2 for Redundancy

        if self.device.type == "cuda":
            # use bfloat16 for the entire project
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs
            #   refer: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self._predictor.to(self.device)
        torch.set_default_device(self.device)

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

    def _get_image_annotations(self, jobId: int, items: list[CondFlattenItem]) -> dict[int, LabeledShape]:
        """Get job Annotations by cvat api"""
        shape_annotations, track_annotations = [], {}
        for item in items:
            state = item.state
            if state.type == "shape":
                shape_annotations.append(state.id)
                continue
            if state.type == "track":
                track_annotations.update({state.id: {"frame_idx": int(item.frame_idx)}})
                continue
            raise ValueError("anno_type must be either 'shape' or 'track', got %s" % state.type)

        # Loading annotations from cvat-client
        annotations = {}
        annotations_infos, response = self._client.jobs_api.retrieve_annotations(jobId)
        for shape in annotations_infos.shapes:
            if shape["id"] in shape_annotations:
                annotations.update({shape["id"]: shape})
        for track in annotations_infos.tracks:
            if track["id"] in track_annotations:
                frame_idx = track_annotations[track["id"]]["frame_idx"]
                shape = [s for s in track.shapes if s["frame"] <= frame_idx and not s["outside"]].pop(-1)
                annotations.update({track["id"]: shape})
        return annotations

    def _get_image_tensors(self, jobId: int, frame_idxes: list[int]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Get images Tensor (preprocessed image) by cvat api"""
        image_tensors, image_sizes = [], []

        for frame_idx in frame_idxes:
            buffer, response = self._client.jobs_api.retrieve_data(
                id=jobId, type="frame", number=frame_idx, quality="original"
            )

            image = Image.open(buffer)
            imgsz = list(image.size[::-1])  # wh -> hw
            image_tensor = self._transform(image)

            # Try to clean image temp file
            buffer.close()
            try:
                os.remove(buffer.name)
            except PermissionError:
                warnings.warn("PermissionError: fail to delete cvat temp image file %s." % buffer.name)

            image_sizes.append(imgsz)
            image_tensors.append(image_tensor)

        image_tensors = torch.stack(image_tensors)

        return image_tensors, image_sizes

    def _get_image_feature_cache(self, jobId: int, frame_idxes: list[int], caches: dict[int, ImageFeature]) -> dict[int, ImageFeature]:
        # Check reusable and invalid from history cache
        cache_frame_idxes = set(caches.keys())
        remove_idxes = cache_frame_idxes - set(frame_idxes) # invalid cache
        increase_idxes = sorted(set(frame_idxes) - cache_frame_idxes)  # reusable cache
        caches = {k: v for k, v in caches.items() if k not in remove_idxes}  # free invalid cache resource
        keys = [REDIS_TMPL_KEY_IMAGE_FEATS.format(jobId=jobId, frame_idx=frame_idx) for frame_idx in increase_idxes]
        caches.update(dict(zip(frame_idxes, self._redis.load_image_feature(keys, self.device))))

        # Get image feature from inferencing image-encoder
        non_cache_idxes = [i for i, v in enumerate(caches.values()) if v is None]
        if len(non_cache_idxes) > 0:  # no_cache for image feats
            non_cache_frame_idxes = [frame_idxes[i] for i in non_cache_idxes]
            image_tensors, image_sizes = self._get_image_tensors(jobId, non_cache_frame_idxes)
            with torch.inference_mode():
                backbone_out = self._predictor.forward_image(image_tensors.to(self.device))
            _, vision_feats, vision_pos_embeds, feat_sizes = self._predictor._prepare_backbone_features(backbone_out)

            # save image feats cache
            non_cache_keys = [keys[i] for i in non_cache_idxes]
            non_cache_vals = (vision_feats, vision_pos_embeds, feat_sizes, image_sizes, )
            self._redis.save_image_feature(non_cache_keys, data=non_cache_vals)

            caches.update({
                frame_idx: ImageFeature.model_validate({
                    "vision_feats": [t[:, j: j + 1, :] for t in vision_feats],
                    "vision_pos_embeds": [t[:, j: j + 1, :] for t in vision_pos_embeds],
                    "feat_sizes": feat_sizes,
                    "imgsz": image_sizes[j],
                    "backbone_out": ImageEmbeddingOutput.model_validate({
                        "vision_features": backbone_out["vision_features"][j: j + 1, ...],
                        "vision_pos_enc": [t[j: j + 1, ...] for t in backbone_out["vision_pos_enc"]],
                        "backbone_fpn": [t[j: j + 1, ...] for t in backbone_out["backbone_fpn"]],
                    }),
                }) for j, frame_idx in enumerate(non_cache_frame_idxes)
            })
        return caches

    def _get_batch_cond_memory_output(self, jobId: int, batch: list[TrackState], batch_size=BATCHSIZE) -> list[TrackStateWithCondMemory]:
        """batch process cond memory output"""
        batch_results = [b.model_dump() for b in batch]  # new TrackStateWithCondMemory batch results

        cond_flatten = [
            CondFlattenItem(batch_idx=i, frame_idx=int(f), objectId=b.objectId, state=v)
            for i, b in enumerate(batch) for f, v in b.cond.items()
        ]

        non_cache_cond_flatten: list[CondFlattenItem] = []
        iters_num = (len(cond_flatten) + batch_size - 1) // batch_size
        for i in range(iters_num):   # iter check redis memory output
            batch_cond_flatten = cond_flatten[i * batch_size: (i + 1) * batch_size]
            batch_keys = [REDIS_TMPL_KEY_MEMORY_OUTS.format(
                jobId=jobId, frame_idx=cond_item.frame_idx, objectId=cond_item.objectId
            ) for cond_item in batch_cond_flatten]

            caches = self._redis.load_object_memory(batch_keys, self.device)
            for cond_item, memory_output in zip(batch_cond_flatten, caches):
                if memory_output is None:
                    non_cache_cond_flatten.append(cond_item)
                else:
                    batch_results[cond_item.batch_idx]["cond"][str(cond_item.frame_idx)] = memory_output

        if len(non_cache_cond_flatten) > 0:  # make cond memory output
            # group non_cache_cond_flatten to accelerate process, reduce inference/cache image features
            non_cache_cond_flatten = group_frame(non_cache_cond_flatten, key=lambda item: item.frame_idx)

            # load annotations like shapes: [x, y, x, y, ...] from cvat api
            annotations = self._get_image_annotations(jobId, non_cache_cond_flatten)

            images_feats_cache = {}
            iters_num = (len(non_cache_cond_flatten) + batch_size - 1) // batch_size
            for i in range(iters_num):  # memory encoder process
                batch_non_cache_cond_flatten = non_cache_cond_flatten[i * batch_size: (i + 1) * batch_size]

                frame_idxes = sorted(set(item.frame_idx for item in batch_non_cache_cond_flatten))
                images_feats_cache = self._get_image_feature_cache(jobId, frame_idxes, images_feats_cache)

                # prepare simulate mask output
                cat_l_masks, cat_h_masks, obj_ptrs, object_score_logits = [], [], [], []
                for cond_item in batch_non_cache_cond_flatten:
                    shape = annotations[cond_item.state.id]
                    imgsz = images_feats_cache[cond_item.frame_idx].imgsz

                    mask = np.zeros(imgsz, dtype=np.uint8)
                    if str(shape.type) == "polygon":
                        cv2.drawContours(mask, [np.array(shape.points, np.int32).reshape(-1, 2)], -1, 1, cv2.FILLED)
                    elif str(shape.type) == "mask":  # RLE decode
                        x0, y0, x1, y1 = map(int, shape.points[-4:])  # shape.points = [rle] + [x0, y0, x1, y1]
                        mask_part = [v for i, n in enumerate(shape.points[:-4]) for v in [int(i) % 2] * int(n)]
                        mask_part = np.array(mask_part).reshape((y1 - y0 + 1, x1 - x0 + 1))
                        mask[y0: y1 + 1, x0: x1 + 1] = mask_part
                    else:
                        raise TypeError("SAM2.1 AutoTrack not supported type: %s" % shape["type"])
                    mask = torch.from_numpy(mask).to(device=self.device, dtype=torch.float)[None, None, ...]

                    # reference: https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py#L415
                    # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
                    out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05; sigmoid(10.0)=1.0000

                    h_res_mask = F.interpolate(
                        mask * out_scale + out_bias,
                        mode="bilinear",
                        align_corners=False,
                        size=(1024, 1024),
                        antialias=True
                    )

                    l_res_mask = F.interpolate(
                        mask * out_scale + out_bias,
                        mode="bilinear",
                        align_corners=False,
                        size=(256,256),
                        antialias=True
                    )

                    cat_l_masks.append(l_res_mask.to(self.device))
                    cat_h_masks.append(h_res_mask.to(self.device))

                    with torch.inference_mode():
                        mask_inputs_float = F.interpolate(
                            mask,
                            mode="bilinear",
                            align_corners=False,
                            size=(1024, 1024),
                            antialias=True
                        )

                        _, _, _, _, _, obj_ptr, _ = self._predictor._forward_sam_heads(
                            backbone_features=images_feats_cache[cond_item.frame_idx].backbone_out.vision_features,
                            mask_inputs=self._predictor.mask_downsample(mask_inputs_float.to(self.device)),
                            high_res_features=images_feats_cache[cond_item.frame_idx].backbone_out.backbone_fpn[:2],
                        )

                    is_obj_appearing = torch.any(mask.flatten(1).float() > 0.0, dim=1)
                    is_obj_appearing = is_obj_appearing[..., None]
                    lambda_is_obj_appearing = is_obj_appearing.float()
                    object_score_logit = out_scale * lambda_is_obj_appearing + out_bias
                    if self._predictor.pred_obj_scores:
                        if self._predictor.fixed_no_obj_ptr:
                            obj_ptr = lambda_is_obj_appearing * obj_ptr
                        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self._predictor.no_obj_ptr

                    obj_ptrs.append(obj_ptr)
                    object_score_logits.append(object_score_logit)

                # Inference memory encoder
                vision_feats = [
                    torch.cat(fpni, dim=1)
                    for fpni in zip(
                        *[
                            images_feats_cache[cond_item.frame_idx].vision_feats
                            for cond_item in batch_non_cache_cond_flatten
                        ]
                    )
                ]
                h_res_masks = torch.cat(cat_h_masks, dim=0)
                l_res_masks = torch.cat(cat_l_masks, dim=0)
                object_score_logits = torch.cat(object_score_logits, dim=0)
                obj_ptrs = torch.cat(obj_ptrs, dim=0)

                with torch.inference_mode():
                    maskmem_features, maskmem_pos_enc = self._predictor._encode_new_memory(
                        current_vision_feats=vision_feats,
                        feat_sizes=FEAT_SIZES,
                        pred_masks_high_res=h_res_masks.to(self.device),
                        object_score_logits=object_score_logits.to(self.device),
                        is_mask_from_pts=True
                    )

                # save memory output cache
                batch_keys = [REDIS_TMPL_KEY_MEMORY_OUTS.format(
                    jobId=jobId, frame_idx=cond_item.frame_idx, objectId=cond_item.objectId
                ) for cond_item in batch_non_cache_cond_flatten]
                batch_vals = (maskmem_features, maskmem_pos_enc, l_res_masks, )
                self._redis.save_object_memory(batch_keys, data=batch_vals)

                # save obj ptr cache
                batch_keys = [REDIS_TMPL_KEY_OBJ_PTR.format(
                    jobId=jobId, frame_idx=cond_item.frame_idx, objectId=cond_item.objectId
                ) for cond_item in batch_non_cache_cond_flatten]
                self._redis.save_image_obj_ptr(batch_keys, data=obj_ptrs)

                for j, cond_item in enumerate(batch_non_cache_cond_flatten):
                    batch_results[cond_item.batch_idx]["cond"][str(cond_item.frame_idx)] = MemoryOutput(
                        maskmem_features=maskmem_features[j: j + 1, ...],
                        maskmem_pos_enc=[pos_enc[j: j + 1, ...] for pos_enc in maskmem_pos_enc],
                        maskmem_pred_masks=cat_l_masks[j]
                    )

        return [TrackStateWithCondMemory.model_validate(batch_result) for batch_result in batch_results]

    def _prepare_batch_mem_pix_feature(self, jobId: int, state: TrackStateWithCondMemory, frame_feature: ImageFeature):
        """reference: sam2/modeling/sam2_base.py function: SAM2Base._prepare_memory_conditioned_features"""
        frame_idx = state.frame
        num_frames = len(state.cond) + len(state.non_cond) + 1
        cond = {int(k): v for k, v in state.cond.items()}    # guarantee dict keys is int
        non_cond = {int(fid): {} for fid in state.non_cond}  # guarantee dict keys is int

        # backbone_out, vision_feats, vision_pos_embeds, feat_sizes, imgsz = frame_feature

        track_in_reverse = frame_idx < sorted([*cond.keys(), *non_cond.keys()])[0]
        B, C, H, W = frame_feature.backbone_out.backbone_fpn[-1].shape
        if self._predictor.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = frame_feature.backbone_out.backbone_fpn[-1]
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Retrieve the memories encoded with the maskmem backbone
        to_cat_memory, to_cat_memory_pos_embed = [], []
        # Add conditioning frame's output first (all cond frames have t_pos=0 for
        # when getting temporal positional embedding below)
        assert len(cond) > 0, "SAM2.1 requires at least one condition to track by memory."
        # Select a maximum number of temporally closest cond frames for cross attention
        selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
            frame_idx, cond, self._predictor.max_cond_frames_in_attn)
        t_pos_and_prevs = [(0, {"frame": i, "memout": out}) for i, out in selected_cond_outputs.items()]
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
            if isinstance(out, MemoryOutput):
                t_pos_and_prevs.append((t_pos, {"frame": prev_frame_idx, "memout": out}))
            elif isinstance(out, dict):
                t_pos_and_prevs.append((t_pos, {"frame": prev_frame_idx, **out}))
            else:
                t_pos_and_prevs.append((t_pos, None))

        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue  # skip padding frames
            elif isinstance(prev.get("memout"), MemoryOutput):
                prev = prev.get("memout")  # MemoryOutput already
            else:
                # Non-cond frames must have cached states.
                # If the cache has expired, tracking must restart from the beginning.
                prev = self._redis.load_object_memory([REDIS_TMPL_KEY_MEMORY_OUTS.format(
                    jobId=jobId, frame_idx=prev["frame"], objectId=state.objectId
                )], self.device)[0]
                assert isinstance(prev, MemoryOutput), (
                    "jobId=%d non_cond cache broken, please track again from the begin!" % jobId)
            feats = prev.maskmem_features.to(self.device, non_blocking=True)
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
            # Temporal positional encoding
            maskmem_enc = prev.maskmem_pos_enc[-1].to(self.device)
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
            maskmem_enc = maskmem_enc + self._predictor.maskmem_tpos_enc[self._predictor.num_maskmem - t_pos - 1]
            to_cat_memory_pos_embed.append(maskmem_enc)

        # Construct the list of past object pointers
        if self._predictor.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = min(num_frames, self._predictor.max_obj_ptrs_in_encoder)
            # First add those object pointers from selected conditioning frames
            # (optionally, only include object pointers in the past during evaluation)
            if self._predictor.only_obj_ptrs_in_the_past_for_eval:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs

            pos_and_ptrs = []
            for t, out in ptr_cond_outputs.items():
                key = REDIS_TMPL_KEY_OBJ_PTR.format(jobId=jobId, frame_idx=t, objectId=state.objectId)
                obj_ptr_cache = self._redis.load_image_obj_ptr([key], self.device)
                if obj_ptr_cache:
                    pos_and_ptrs.append((
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self._predictor.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        obj_ptr_cache[0]
                    ))

            # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or t >= num_frames:
                    break
                out = non_cond.get(t, unselected_cond_outputs.get(t, None))
                if out is not None:
                    key = REDIS_TMPL_KEY_OBJ_PTR.format(jobId=jobId, frame_idx=t, objectId=state.objectId)
                    obj_ptr_cache = self._redis.load_image_obj_ptr([key], self.device)
                    if obj_ptr_cache:
                        pos_and_ptrs.append((t_diff, obj_ptr_cache[0]))
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
                    obj_pos = torch.tensor(pos_list).to(device=self.device, non_blocking=True)
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

        with torch.inference_mode():
            pix_feat_with_mem = self._predictor.memory_attention(
                curr=frame_feature.vision_feats[-1],
                curr_pos=frame_feature.vision_pos_embeds[-1],
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens
            )

        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _batch_step(self, jobId: int, items: list[TrackState], batch_size=BATCHSIZE):
        images_feats_cache = {}  # batch load image feats for cache
        # group frames for reducing inference image encoder module repeatedly
        items = group_frame(items, key=lambda item: item.frame)
        iters_num = (len(items) + batch_size - 1) // batch_size
        for i in range(iters_num):
            batch = items[i * batch_size: (i + 1) * batch_size]

            # prepare cond memory output first, it can accelerate inference for the first track
            batch = self._get_batch_cond_memory_output(jobId, batch, batch_size)

            frame_idxes = sorted(set(state.frame for state in batch))
            images_feats_cache = self._get_image_feature_cache(jobId, frame_idxes, images_feats_cache)

            batch_mem_pix_feat = torch.cat([
                self._prepare_batch_mem_pix_feature(
                    jobId, state, images_feats_cache[state.frame]
                ) for state in batch
            ])

            with torch.inference_mode():
                high_res_features = [
                    torch.cat(batch_high_res_feature)
                    for batch_high_res_feature in zip(
                        *[
                            images_feats_cache[state.frame].backbone_out.backbone_fpn[:2]
                            for state in batch
                        ]
                    )
                ]

                (
                    l_res_multimasks,
                    h_res_multimasks,
                    ious,
                    l_res_masks,
                    h_res_masks,
                    obj_ptrs,
                    object_score_logits,
                ) = self._predictor._forward_sam_heads(
                    backbone_features=batch_mem_pix_feat,
                    high_res_features=high_res_features,
                )

                batch_vision_feats = [
                    torch.cat(vision_feats, dim=1)
                    for vision_feats in zip(
                        *[images_feats_cache[state.frame].vision_feats for state in batch]
                    )
                ]

                maskmem_features, maskmem_pos_enc = self._predictor._encode_new_memory(
                    current_vision_feats=batch_vision_feats,
                    feat_sizes=FEAT_SIZES,
                    pred_masks_high_res=h_res_masks,
                    object_score_logits=object_score_logits,
                    is_mask_from_pts=False
                )

            # save memory output cache
            batch_keys = [REDIS_TMPL_KEY_MEMORY_OUTS.format(
                jobId=jobId, frame_idx=state.frame, objectId=state.objectId
            ) for state in batch]
            batch_vals = (maskmem_features, maskmem_pos_enc, l_res_masks, )
            self._redis.save_object_memory(batch_keys, data=batch_vals)

            # save obj ptr cache
            batch_keys = [REDIS_TMPL_KEY_OBJ_PTR.format(
                jobId=jobId, frame_idx=state.frame, objectId=state.objectId
            ) for state in batch]
            self._redis.save_image_obj_ptr(batch_keys, obj_ptrs)

            for bn, state in enumerate(batch):
                # confidence, mask_tensor(Batch, 1, 256, 256), origin_img_size(h, w), state
                yield ious[bn].squeeze(-1), l_res_masks[bn: bn + 1], images_feats_cache[state.frame].imgsz, state

    def handle(
        self,
        jobId: int,
        items: list[TrackState],
        threshold: float = THRESHOLD,
        approx_threshold: float = APPROX_THRESHOLD,
        batch_size=BATCHSIZE
    ):
        items = list(map(TrackState.model_validate, items))
        for confidence, mask_tensor, img_size, state in self._batch_step(jobId, items, batch_size):
            if confidence < threshold:
                yield None
                continue

            mask = F.interpolate(mask_tensor, mode="bilinear", align_corners=False, size=img_size)
            mask = mask.cpu().numpy()[0, 0]  # drop (batch, channel) dims
            mask = (mask > 0).astype(np.uint8) * 255
            if mask[mask > 0].size <= 4:
                yield None
                continue

            if state.type == "polygon":
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                points = contours[int(np.argmax([cv2.contourArea(c) for c in contours]))]
                points = cv2.approxPolyDP(points, approx_threshold, True)
                points = points.reshape(-1).tolist()
                yield {"confidence": float(confidence), "labelId": state.labelId, "points": points, "type": state.type}
            elif state.type == "mask":
                coords = np.where(mask > 0)
                y0, x0 = coords[0].min(), coords[1].min()
                y1, x1 = coords[0].max(), coords[1].max()
                flat = mask[y0: y1, x0: x1].ravel()
                (run_indices,) = np.diff(flat, prepend=[not flat[0]], append=[not flat[-1]]).nonzero()
                run_lengths = np.diff(run_indices, prepend=[0]) if flat[0] else np.diff(run_indices)
                points = run_lengths.tolist() + list(map(int, [x0, y0, x1 - 1, y1 - 1]))
                yield {"confidence": float(confidence), "labelId": state.labelId, "points": points, "type": state.type}
            else:
                raise NotImplementedError("type %s not support SAM2.1 AutoTrack." % state.type)


            yield {"confidence": float(conf), "labelId": info["labelId"], "points": points, "type": info["type"]}
