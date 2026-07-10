# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import io
import time
import torch
import pickle
import numpy as np
import contextlib

from collections.abc import Callable


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


def torch_to_device(data: torch.Tensor | dict | list | tuple, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    if isinstance(data, dict):
        return {k: torch_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [torch_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(torch_to_device(v, device) for v in data)
    return data


def torch_to_cpu(data: torch.Tensor | dict | list | tuple):
    if isinstance(data, torch.Tensor):
        return data.detach().to("cpu")
    if isinstance(data, dict):
        return {k: torch_to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [torch_to_cpu(v) for v in data]
    if isinstance(data, tuple):
        return tuple(torch_to_cpu(v) for v in data)
    return data


def torch_loads(content, device: torch.device):
    buffer = io.BytesIO(content)
    data = torch.load(buffer, map_location="cpu", weights_only=False)
    return torch_to_device(data, device)


def torch_dumps(data: torch.Tensor | dict | list | tuple) -> bytes:
    buffer = io.BytesIO()
    torch.save(torch_to_cpu(data), buffer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return buffer.getvalue()


class Profile(contextlib.ContextDecorator):
    """Context manager and decorator for profiling code execution time, with optional CUDA synchronization."""

    def __init__(self, t=0.0, device: torch.device = None):
        self.t, self.device, self.cuda = t, device, bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Initializes timing at the start of a profiling context block for performance measurement."""
        self.start = self.time()
        return self

    def __exit__(self, type_, value, traceback):
        """Concludes timing, updating duration for profiling upon exiting a context block."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def group_frame(items: list, key: Callable):
    """
    Group the same frame_idx together and order it by len max first.

    :param items: [{"frame": int, ...}, ...]
    :param key: function
    :return: items
    """
    items_order = {}
    for i, item in enumerate(items):
        frame_idx = key(item)
        group = items_order.get(frame_idx, [])
        group.append(i)
        items_order[frame_idx] = group
    ids = sorted(items_order.values(), key=lambda k: len(k), reverse=True)
    return [items[i] for gids in ids for i in gids]
