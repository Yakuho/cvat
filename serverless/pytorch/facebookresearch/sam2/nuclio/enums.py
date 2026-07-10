import torch

from pydantic import BaseModel, ConfigDict


class TensorBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CondState(BaseModel):
    type: str
    id  : int


class CondFlattenItem(BaseModel):
    batch_idx: int
    frame_idx: int
    objectId : str
    state    : CondState


class MemoryOutput(TensorBaseModel):
    maskmem_features  : torch.Tensor
    maskmem_pos_enc   : list[torch.Tensor]
    maskmem_pred_masks: torch.Tensor


class ImageEmbeddingOutput(TensorBaseModel):
    backbone_fpn   : list[torch.Tensor]
    vision_pos_enc : list[torch.Tensor]
    vision_features: torch.Tensor


class ImageFeature(TensorBaseModel):
    backbone_out     : ImageEmbeddingOutput
    vision_feats     : list[torch.Tensor]
    vision_pos_embeds: list[torch.Tensor]
    feat_sizes       : list
    imgsz            : list


class TrackStateBase(TensorBaseModel):
    frame   : int
    objectId: str
    labelId : int
    type    : str
    non_cond: list[int]


class TrackState(TrackStateBase):
    cond: dict[str, CondState]


class TrackStateWithCondMemory(TrackStateBase):
    cond: dict[str, MemoryOutput]
