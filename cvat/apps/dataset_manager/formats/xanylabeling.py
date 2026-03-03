# Copyright (C) 2026 Intel Corporation
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import os
import json
import cv2
import numpy as np

from pathlib import Path
from pyunpack import Archive
from cvat.apps.dataset_manager.util import make_zip_archive
from cvat.apps.dataset_manager.formats.cvat import dump_media_files
from cvat.apps.dataset_manager.bindings import TaskData
from .registry import dm_env, exporter, importer


class Dataset:
    """This Object is used to capability of datumaro.Dataset, but not full-match yet temporary"""
    def __init__(self, temp_dir: str):
        # name mapping for fixing string decode error when contain like chinese characters
        self.temp_dir = temp_dir
        self._files = dict((
            str(f)[len(temp_dir) + 1:].encode("cp437").decode("gbk").rsplit('.', 1)[0], f)
            for f in Path(temp_dir).glob("**/*.json"))

    def get_annatations(self, filepath: str):
        """get xanylabeling shapes from json file (notice: no suffix)"""
        annatations = []
        if filepath in self._files:
            with open(self._files[filepath], encoding="utf-8") as f:
                annatations = json.load(f)["shapes"]
        return annatations


class XAnyLabelingBase:
    _TYPE_MAP = {"polyline": "linestrip", "box": "rectangle", "points": "point"}
    _SUPPORT_TYPES = ("box", "polygon", "rectangle", "polyline", "points")  # unsupported: ellipse, cuboid, mask

    def _export_shapes_iters(self, frame_annotation):
        """get xanylabeling shapes from cvat frame_annotation"""
        groups_id = {}  # record shape type group id for capability
        for shape in frame_annotation.labeled_shapes:
            if shape.type not in self._SUPPORT_TYPES:
                continue

            if shape.type in ("box", "rectangle"):
                x0, y0, x1, y1 = list(shape.points)
                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            else:
                it = iter(shape.points)
                points = [list(pair) for pair in zip(it, it)]

            shape_base = {
                "kie_linking": [],
                "label": shape.label,
                "score": None,
                "points": points,
                "group_id": str(shape.group) if shape.group else None,  # default=0
                "description": "",
                "difficult": False,
                "shape_type": self._TYPE_MAP.get(
                    shape.type, shape.type),
                "flags": {},
                "attributes": {}
            }

            if shape.type == "points" and len(points) > 1:
                group_id = groups_id.get(shape.type, 0)
                groups_id[shape.type] = group_id + 1
                group_name = "points-{}".format(group_id)
                if isinstance(shape_base["group_id"], str):
                    group_name = "{}-{}".format(shape_base["group_id"], group_name)
                shape_base.update({"group_id": group_name})
                for point in points:
                    yield {**shape_base, "points": [point]}
            elif shape.type in ("box", "rectangle") and shape.rotation != 0:
                center_point = np.mean(points, axis=0)
                # negative angle for opencv clockwise and cvat angle is positive angle absolutaly
                m = cv2.getRotationMatrix2D(tuple(center_point), -shape.rotation, 1.0)
                shape_base.update({
                    "shape_type": "rotation",
                    "direction": np.deg2rad(shape.rotation),
                    "points": cv2.transform(np.array(points).reshape(-1, 1, 2), m).reshape(-1, 2).tolist()
                })
                yield shape_base
            else:
                yield shape_base

    def _export(self, temp_dir: str, instance_data: TaskData):
        """convert format cvat to xanylabeling"""
        for frame_annotation in instance_data.group_by_frame(include_empty=True):
            savefile = Path(temp_dir) / (frame_annotation.name.rsplit(".", 1)[0] + ".json")
            os.makedirs(str(savefile.parent), exist_ok=True)
            with open(savefile, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "version": self.VERSION,
                    "flags": {},
                    "shapes": list(self._export_shapes_iters(frame_annotation)),
                    "imagePath": frame_annotation.name,
                    "imageData": None,
                    "imageHeight": int(frame_annotation.height),
                    "imageWidth": int(frame_annotation.width),
                    "description": ""
                }, ensure_ascii=False, indent=2))

    def _import_shapes_iters(self, shapes):
        """get cvat frame_annotation from xanylabeling shapes"""
        points_shape, groups = dict(), dict()

        # binding point shapes to cvat points when group_id the same
        for _ in range(len(shapes)):
            shape = shapes.pop(0)
            if shape["shape_type"] == "point":
                record = points_shape.get(shape["group_id"], [])
                record.append(shape)
                points_shape.update({shape["group_id"]: record})
            else:
                shapes.append(shape)
        for group_id, points_group_shape in points_shape.items():
            if group_id is None:  # no group
                for shape in points_group_shape:
                    shapes.append(shape)
            else:
                shape1st = {**points_group_shape[0]}  # 1st shape for template
                shape1st.update({"points": [], "attributes": {}})
                for shape in points_group_shape:
                    shape1st["points"] += shape["points"]
                if len(points_group_shape) > 1:
                    shape1st.update({"group_id": None})
                shapes.append(shape1st)

        for shape in shapes:
            if shape["shape_type"] in ("rectangle", "rotation"):
                points = shape["points"]
                assert len(points) in (2, 4), "rectangle shape must have 2 or 4 points"
                if len(points) == 4:  # convert (left-top, right-top, right-bot, left-bot) to (left-top, right-bot)
                    points = [points[i] for i in [0, 2]]
                if shape.get("direction"):
                    # negative angle for opencv clockwise and xanylabeling angle is positive angle absolutaly
                    # xanylabeling rotation boxes xys is (rotation boxes), but cvat rotation boxes xys is (no rotation boxes)
                    # so rotate xanylabeling boxes xys to no rotation boxes
                    center_point = np.mean(points, axis=0)
                    rotation = np.rad2deg(shape["direction"])
                    m = cv2.getRotationMatrix2D(tuple(center_point), rotation, 1.0)
                    points = cv2.transform(np.array(points).reshape(-1, 1, 2), m).reshape(-1, 2).tolist()
            elif shape["shape_type"] == "circle":
                ((cx, cy), (rx, ry)) = shape["points"]
                radius = float(np.linalg.norm([rx - cx, ry - cy]))
                points = [[cx, cy], [cx + radius, cy - radius]]
            else:
                points = shape["points"]

            group = 0
            if shape["group_id"] is not None:
                group = groups.get(shape["group_id"], len(groups)) + 1
                groups[shape["group_id"]] = group

            yield dict(
                type=self._TYPE_MAP.get(shape["shape_type"], shape["shape_type"]),
                rotation=round(np.rad2deg(shape["direction"]), 2) if shape.get("direction") else 0.0,
                label=shape["label"],
                points=[v for xy in points for v in xy],  # flatten points from [[x, y], ...]
                attributes=[],
                group=group
            )

    def _import(self, dataset: Dataset, instance_data: TaskData):
        """convert format xanylabeling to cvat"""
        for frame_annotation in instance_data.group_by_frame(include_empty=True):
            shapes = dataset.get_annatations(frame_annotation.name.rsplit('.', 1)[0])  # query by full filepath with no suffix
            for parsed_shape in self._import_shapes_iters(shapes):
                instance_data.add_shape(instance_data.LabeledShape(
                    **parsed_shape,
                    z_order=0,
                    source="auto",
                    occluded=False,
                    frame=frame_annotation.frame,
                ))


@exporter(name="XAnyLabeling", ext="ZIP", version="3.3.1")
class XAnyLabelingExporter(XAnyLabelingBase):
    """
    XAnyLabelingExporter will be set attributes by exporter decorator:
        self.NAME: XAnyLabeling
        self.VERSION: 3.3.1
        self.EXT: ZIP
        self.DISPLAY_NAME: ...
        self.DIMENSION: ...
        self.ENABLED: ...
    """
    def __call__(self, dst_file, temp_dir: str, instance_data: TaskData, save_images: bool = False):
        if save_images:
            dump_media_files(instance_data, temp_dir)
        self._export(temp_dir, instance_data)
        make_zip_archive(temp_dir, dst_file)


@importer(name="xanylabeling", ext="ZIP", version="3.3.1")
class XAnyLabelingImporter(XAnyLabelingBase):
    """
    XAnyLabelingImporter will be set attributes by importer decorator:
        self.NAME: XAnyLabeling
        self.VERSION: 3.3.1
        self.EXT: ZIP
        self.DISPLAY_NAME: ...
        self.DIMENSION: ...
        self.ENABLED: ...
    """

    _TYPE_MAP = {
        **{v: k for k, v in XAnyLabelingBase._TYPE_MAP.items()},  # reverse kv
        "circle": "ellipse", "rectangle": "rectangle", "rotation": "rectangle"
    }

    def __call__(self, src_file, temp_dir: str, instance_data: TaskData, load_data_callback=None, **kwargs):
        # mask2poly = kwargs.get("conv_mask_to_poly", False)
        Archive(src_file.name).extractall(temp_dir)
        dataset = Dataset(temp_dir)
        # dataset = MaskToPolygonTransformation.convert_dataset(dataset, **kwargs)
        # if load_data_callback is not None:
        #     load_data_callback(dataset, instance_data)
        self._import(dataset, instance_data)
