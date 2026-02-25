# Copyright (C) 2026 Intel Corporation
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import lxml.etree as ET
import json

from pathlib import Path
from cvat.apps.dataset_manager.util import make_zip_archive
from cvat.apps.dataset_manager.formats.cvat import dump_media_files, dump_task_or_job_anno, dump_as_cvat_annotation
from .registry import dm_env, exporter, importer


class CVATAnnotation:
    def __init__(self, path, dst=None, version="3.3.1"):
        """

        :param path: cvat annotation xml
        :param dst:  save dir
        """
        tree = ET.parse(path)
        self.version = version
        self.root = tree.getroot()
        self.dst = dst
        if self.dst is None:
            self.dst = str(Path(path).parent)

    def toxanylabeling(self):
        _TYPE_MAP = {"polyline": "linestrip", "box": "rectangle", "points": "point"}

        for annotation in self.root.findall("image"):
            shapes = []
            groups_id = [0]  # points group  TODO: Not fully compatible
            for shape in annotation.findall("*"):
                if shape.tag == "box":
                    x0, y0, x1, y1 = shape.attrib["xtl"], shape.attrib["ytl"], shape.attrib["xbr"], shape.attrib["ybr"]
                    x0, y0, x1, y1 = list(map(float, [x0, y0, x1, y1]))
                    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                elif shape.tag == "ellipse":
                    continue # xanylabeling not support ellipse default
                else:
                    points = [list(map(float, xy.split(","))) for xy in shape.attrib["points"].split(";")]

                if shape.tag == "points":
                    group_id = None
                    if len(points)> 1:
                        group_id = groups_id[0]
                        groups_id[0] = groups_id[0] + 1
                    for point in points:
                        shapes.append({
                            "kie_linking": [],
                            "label": shape.attrib["label"],
                            "score": None,
                            "points": [point],
                            "group_id": group_id,
                            "description": "",
                            "difficult": False,
                            "shape_type": _TYPE_MAP.get(shape.tag, shape.tag),
                            "flags": {},
                            "attributes": {}
                        })
                else:
                    shapes.append({
                        "kie_linking": [],
                        "label": shape.attrib["label"],
                        "score": None,
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": _TYPE_MAP.get(shape.tag, shape.tag),
                        "flags": {},
                        "attributes": {}
                    })

            label_data = {
                "version": self.version,
                "flags": {},
                "shapes": shapes,
                "imagePath": annotation.attrib["name"],
                "imageData": None,
                "imageHeight": int(annotation.attrib["height"]),
                "imageWidth": int(annotation.attrib["width"]),
                "description": ""
            }

            savefile = Path(self.dst) / (annotation.attrib["name"].rsplit(".", 1)[0] + ".json")
            os.makedirs(str(savefile.parent), exist_ok=True)
            with open(savefile, "w", encoding="utf-8") as f:
                f.write(json.dumps(label_data, ensure_ascii=False, indent=2))


def _export_images(dst_file, temp_dir, instance_data, save_images=False):
    # create cvat annotation xml
    annotation_file = osp.join(temp_dir, "annotations.xml")
    with open(annotation_file, "wb") as f:
        dump_task_or_job_anno(f, instance_data, dump_as_cvat_annotation)

    # convert cvat annotation to xanylabeling format
    CVATAnnotation(annotation_file, temp_dir).toxanylabeling()
    os.remove(annotation_file)

    if save_images:
        dump_media_files(instance_data, temp_dir)

    make_zip_archive(temp_dir, dst_file)


@exporter(name="XAnyLabeling", ext="ZIP", version="3.3.1")
def _export(dst_file, temp_dir, instance_data, save_images=False):
    _export_images(dst_file, temp_dir, instance_data, save_images)


# @importer(name="xanylabeling", ext="ZIP", version="3.x")
# def _import(src_file, temp_dir, instance_data, load_data_callback=None, **kwargs):
#     Archive(src_file.name).extractall(temp_dir)
#
#     detect_dataset(temp_dir, format_name="label_me", importer=dm_env.importers.get("label_me"))
#     dataset = Dataset.import_from(temp_dir, "label_me", env=dm_env)
#     dataset = MaskToPolygonTransformation.convert_dataset(dataset, **kwargs)
#     if load_data_callback is not None:
#         load_data_callback(dataset, instance_data)
#     import_dm_annotations(dataset, instance_data)
