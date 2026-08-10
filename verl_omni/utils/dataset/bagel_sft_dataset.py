# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/ByteDance-Seed/Bagel/tree/main/data


import io
import json
import logging
import os
import random
import re
import subprocess
import traceback

import numpy as np
import pyarrow.fs as pf
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image, ImageFile, PngImagePlugin
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes, strict=False):
        assert attn_mode in ["causal", "full", "noise"]
        if attn_mode == "causal":
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum : csum + s, :csum] = 1
        else:
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
            attention_mask[csum : csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes, strict=False):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(~attention_mask, float("-inf"))
    return attention_mask


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")
    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for _, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []
    if "<|im_start|>" not in all_special_tokens:
        new_tokens.append("<|im_start|>")
    if "<|im_end|>" not in all_special_tokens:
        new_tokens.append("<|im_end|>")
    if "<|vision_start|>" not in all_special_tokens:
        new_tokens.append("<|vision_start|>")
    if "<|vision_end|>" not in all_special_tokens:
        new_tokens.append("<|vision_end|>")

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )
    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction="square"):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14 * 14 * 9 * 1024,
        image_mean=None,
        image_std=None,
    ):
        self.stride = image_stride
        if image_mean is None:
            image_mean = [0.5, 0.5, 0.5]
        if image_std is None:
            image_std = [0.5, 0.5, 0.5]
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def read_frames_pyav(video_path, num_frames, sample="rand", fix_start=None, clip=None, min_num_frames=4):
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV (`pip install av`) is required to decode video files for BAGEL SFT.") from exc

    container = av.open(video_path)
    try:
        if not container.streams.video:
            raise ValueError(f"Video {video_path} contains no video streams.")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 1.0
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    finally:
        container.close()

    if not frames:
        raise ValueError(f"Video {video_path} contains no decodable frames.")

    vlen = len(frames)
    if clip:
        start, end = clip
        duration = end - start
        start_index = int(start * fps)
        end_index = min(start_index + int(duration * fps), vlen)
        frames = frames[start_index:end_index]
        vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    return [frames[i] for i in frame_indices]


def extract_frame_number(filename):
    match = re.search(r"_(\d+).jpg$", filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def read_frames_folder(video_path, num_frames, sample="rand", fix_start=None, min_num_frames=4):
    image_list = sort_frames(list(os.listdir(video_path)))
    frames = []
    for image in image_list:
        fp = os.path.join(video_path, image)
        frame = Image.open(fp).convert("RGB")
        frames.append(frame)
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    return frames


class FrameSampler:
    def __init__(self, max_num_frames=-1, min_num_frames=8, sample="rand"):
        self.max_num_frames = max_num_frames
        self.min_num_frames = min_num_frames
        self.sample = sample

    def __call__(self, file_name):
        fn = read_frames_folder if file_name.endswith("/") else read_frames_pyav
        frames = fn(file_name, num_frames=self.max_num_frames, min_num_frames=self.min_num_frames, sample=self.sample)
        return frames


def get_hdfs_host():
    return "hdfs://xxx"


def get_hdfs_block_size():
    return 134217728


def get_hdfs_extra_conf():
    return None


def hdfs_ls_cmd(dir):
    result = subprocess.run(["hdfs", "dfs", "ls", dir], capture_output=True, text=True).stdout
    return ["hdfs://" + i.split("hdfs://")[-1].strip() for i in result.split("\n") if "hdfs://" in i]


def get_parquet_data_paths(data_dir_list, num_sampled_data_paths, rank=0, world_size=1):
    num_data_dirs = len(data_dir_list)
    if world_size > 1:
        chunk_size = (num_data_dirs + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, num_data_dirs)
        local_data_dir_list = data_dir_list[start_idx:end_idx]
        local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
    else:
        local_data_dir_list = data_dir_list
        local_num_sampled_data_paths = num_sampled_data_paths

    local_data_paths = []
    for data_dir, num_data_path in zip(local_data_dir_list, local_num_sampled_data_paths, strict=False):
        if data_dir.startswith("hdfs://"):
            files = hdfs_ls_cmd(data_dir)
            data_paths_per_dir = [file for file in files if file.endswith(".parquet")]
        else:
            files = os.listdir(data_dir)
            data_paths_per_dir = [os.path.join(data_dir, name) for name in files if name.endswith(".parquet")]
        repeat = num_data_path // len(data_paths_per_dir)
        data_paths_per_dir = data_paths_per_dir * (repeat + 1)
        local_data_paths.extend(data_paths_per_dir[:num_data_path])

    if world_size > 1:
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, local_data_paths)

        combined_chunks = []
        for chunk_list in gather_list:
            if chunk_list is not None:
                combined_chunks.extend(chunk_list)
    else:
        combined_chunks = local_data_paths

    return combined_chunks


def init_arrow_pf_fs(parquet_file_path):
    if parquet_file_path.startswith("hdfs://"):
        fs = pf.HadoopFileSystem(
            host=get_hdfs_host(),
            port=0,
            buffer_size=get_hdfs_block_size(),
            extra_conf=get_hdfs_extra_conf(),
        )
    else:
        fs = pf.LocalFileSystem()
    return fs


class DistributedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, local_rank=0, world_size=1, num_workers=8):
        self.dataset_name = dataset_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.rng = random.Random()
        self.data_paths = None

    def get_data_paths(self, *args, **kwargs):
        raise NotImplementedError

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return

        if isinstance(self.data_paths[0], tuple):
            data_paths = sorted(self.data_paths, key=lambda x: (x[0], x[1]))
        elif isinstance(self.data_paths[0], str):
            data_paths = sorted(self.data_paths)
        else:
            raise ValueError(f"Unknown data_paths type: {type(self.data_paths[0])}")

        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        num_files_per_rank = len(data_paths) // self.world_size
        local_start = self.local_rank * num_files_per_rank
        local_end = (self.local_rank + 1) * num_files_per_rank
        self.num_files_per_rank = num_files_per_rank
        self.data_paths_per_rank = data_paths[local_start:local_end]

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.data_paths_per_rank, 0

        worker_id = info.id
        num_files_per_worker = self.num_files_per_rank // info.num_workers
        start = num_files_per_worker * worker_id
        end = num_files_per_worker * (worker_id + 1)
        data_paths_per_worker = self.data_paths_per_rank[start:end]
        return data_paths_per_worker[::-1], worker_id

    def __iter__(self):
        raise NotImplementedError


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row["image"]
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            except Exception as e:
                                print(f"Error: {e} in rg#{row_group_id}, {parquet_file_path}")
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride**2

                            try:
                                caption_dict = row["captions"]
                                caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                print(f"Error: {e} in rg#{row_group_id}, {parquet_file_path}")
                                continue

                            caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                            if len(caps_token) == 0:
                                print(f"no caption in rg#{row_group_id}, {parquet_file_path}")
                                caption_token = self.tokenizer.encode(" ")
                            else:
                                caption_token = random.choice(caps_token)

                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append(
                                {
                                    "type": "text",
                                    "enable_cfg": 1,
                                    "loss": 0,
                                    "special_token_loss": 0,
                                    "special_token_label": None,
                                }
                            )
                            sequence_plan.append(
                                {
                                    "type": "vae_image",
                                    "enable_cfg": 0,
                                    "loss": 1,
                                    "special_token_loss": 0,
                                    "special_token_label": None,
                                }
                            )

                            sample = dict(
                                image_tensor_list=[image_tensor],
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                },
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class SftJSONLIterableDataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        frame_sampler,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        shuffle_lines,
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(jsonl_path_list, data_dir_list, num_used_data, strict=False):
            with open(jsonl_path) as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        elements = []
        for conversation in data["conversations"]:
            if conversation["from"] == "human":
                if "<image>" not in conversation["value"]:
                    elements.append(
                        {
                            "type": "text",
                            "has_loss": 0,
                            "text": conversation["value"],
                        }
                    )
                else:
                    text_list = conversation["value"].split("<image>")
                    for idx, text in enumerate(text_list):
                        if text.strip() != "":
                            elements.append(
                                {
                                    "type": "text",
                                    "has_loss": 0,
                                    "text": text.strip(),
                                }
                            )
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({"type": "image"})
            elif conversation["from"] == "gpt":
                elements.append(
                    {
                        "type": "text",
                        "has_loss": 1,
                        "text": conversation["value"],
                    }
                )
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    data_item = json.loads(data)
                    raw_images = None
                    if "image" in data_item:
                        if isinstance(data_item["image"], list):
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, image))) for image in data_item["image"]
                            ]
                        else:
                            raw_images = [pil_img2rgb(Image.open(os.path.join(image_dir, data_item["image"])))]
                    elif "video" in data_item:
                        raw_images = self.frame_sampler(os.path.join(image_dir, data_item["video"]))
                        special_tokens = "<image>" * len(raw_images)
                        for item in data_item["conversations"]:
                            if "<video>" in item["value"]:
                                item["value"] = item["value"].replace("<video>", special_tokens)
                                break
                            else:
                                raise ValueError("Cannot find <video> in the conversation!")
                except Exception:
                    traceback.print_exc()
                    continue

                if raw_images:
                    for raw_image in raw_images:
                        image_tensor = self.transform(raw_image, img_num=len(raw_images))
                        image_tensor_list.append(image_tensor)
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // transform_stride**2

                elements = self.change_format(data_item, len(image_tensor_list))
                for item in elements:
                    if item["type"] == "text":
                        text_data = item["text"]
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                "type": "text",
                                "enable_cfg": 0,
                                "loss": item["has_loss"],
                                "special_token_loss": 0,
                                "special_token_label": None,
                            }
                            sequence_plan.append(current_plan)
                    elif item["type"] == "image":
                        current_plan = {
                            "type": "vit_image",
                            "enable_cfg": 0,
                            "loss": 0,
                            "special_token_loss": 0,
                            "special_token_label": None,
                        }
                        sequence_plan.append(current_plan)

                has_loss = [item["loss"] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print("No loss defined, skipped.")
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class InterleavedBaseIterableDataset(DistributedIterableDataset):
    def _init_data(self):
        data = {
            "sequence_plan": [],
            "text_ids_list": [],
            "image_tensor_list": [],
            "num_tokens": 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data["num_tokens"] += len(text_ids)
        data["text_ids_list"].append(text_ids)
        data["sequence_plan"].append(
            {
                "type": "text",
                "enable_cfg": int(enable_cfg),
                "loss": int(need_loss),
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride**2
            data["image_tensor_list"].append(image_tensor)

        if need_vae:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride**2
            data["image_tensor_list"].append(image_tensor.clone())

        if need_vit:
            data["sequence_plan"].append(
                {
                    "type": "vit_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.vit_transform.stride**2
            data["image_tensor_list"].append(vit_image_tensor)

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes, strict=False)):
                current_sequence_plan = {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "split_start": idx == 0,
                    "split_end": idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan["frame_delta"] = frame_indexes[idx + 1] - frame_idx
                data["sequence_plan"].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data["image_tensor_list"].append(image_tensor)
                data["num_tokens"] += width * height // self.transform.stride**2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes, strict=False)):
                current_sequence_plan = {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "split_start": idx == 0,
                    "split_end": idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan["frame_delta"] = frame_indexes[idx + 1] - frame_idx
                data["sequence_plan"].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data["image_tensor_list"].append(image_tensor)
                data["num_tokens"] += width * height // self.transform.stride**2

        return data


class ParquetStandardIterableDataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        data_dir_list,
        num_used_data,
        parquet_info,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, parquet_info)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data, strict=False):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                if data_path in parquet_info.keys():
                    num_row_groups = parquet_info[data_path]["num_row_groups"]
                    for rg_idx in range(num_row_groups):
                        row_groups.append((data_path, rg_idx))
        return row_groups

    def parse_row(self, row):
        raise NotImplementedError

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            global_row_group_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at global_rg#{global_row_group_start_id}, row#{row_start_id}"
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    try:
                        fr = pq.ParquetFile(f)
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]
                    except Exception as e:
                        print(f"Error {e} in rg#{row_group_id}, {parquet_file_path}")
                        continue

                    for row_idx, row in df.iterrows():
                        try:
                            data = self.parse_row(row)
                            if len(data) == 0:
                                continue
                            data["data_indexes"] = {
                                "data_indexes": [global_row_group_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        except Exception as e:
                            print(f"Error {e} in rg#{row_group_id}, {parquet_file_path}")
                            continue
                        yield data

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def parse_row(self, row):
        image_num = len(row["image_list"])
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )

        if end_idx - start_idx > 1 and random.random() < 0.5:
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx - 1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx - 1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=True,
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )
        return data


DATASET_REGISTRY = {
    "t2i_pretrain": T2IIterableDataset,
    "vlm_sft": SftJSONLIterableDataset,
    "unified_edit": UnifiedEditIterableDataset,
}


DATASET_INFO = {
    "t2i_pretrain": {
        "t2i": {
            "data_dir": "your_data_path/bagel_example/t2i",
            "num_files": 10,
            "num_total_samples": 1000,
        },
    },
    "unified_edit": {
        "seedxedit_multi": {
            "data_dir": "your_data_path/bagel_example/editing/seedxedit_multi",
            "num_files": 10,
            "num_total_samples": 1000,
            "parquet_info_path": "your_data_path/bagel_example/editing/parquet_info/seedxedit_multi.json",
        },
    },
    "vlm_sft": {
        "llava_ov": {
            "data_dir": "your_data_path/bagel_example/vlm/images",
            "jsonl_path": "your_data_path/bagel_example/vlm/llava_ov_si.jsonl",
            "num_total_samples": 1000,
        },
    },
}


class DataConfig:
    def __init__(
        self,
        grouped_datasets,
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_config,
        tokenizer,
        special_tokens,
        local_rank,
        world_size,
        num_workers,
        expected_num_tokens=32768,
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(data_config.grouped_datasets, data_status)
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop("is_mandatory", False))
            grouped_weights.append(dataset_args.pop("weight", 0.0))

            if "frame_sampler_args" in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop("frame_sampler_args"))
                dataset_args["frame_sampler"] = frame_sampler
            if "image_transform_args" in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop("image_transform_args"))
                dataset_args["transform"] = transform
            if "vit_image_transform_args" in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop("vit_image_transform_args"))
                dataset_args["vit_transform"] = vit_transform

            assert "dataset_names" in dataset_args.keys()
            dataset_names = dataset_args.pop("dataset_names")
            dataset_args["data_dir_list"] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f"Preparing Dataset {grouped_dataset_name}/{item}")
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args["data_dir_list"].append(meta_info["data_dir"])

                if "parquet_info_path" in meta_info.keys():
                    if "parquet_info" not in dataset_args.keys():
                        dataset_args["parquet_info"] = {}
                    with open(meta_info["parquet_info_path"]) as f:
                        parquet_info = json.load(f)
                    dataset_args["parquet_info"].update(parquet_info)

                if "json_dir" in meta_info.keys():
                    if "json_dir_list" not in dataset_args.keys():
                        dataset_args["json_dir_list"] = [meta_info["json_dir"]]
                    else:
                        dataset_args["json_dir_list"].append(meta_info["json_dir"])

                if "jsonl_path" in meta_info.keys():
                    if "jsonl_path_list" not in dataset_args.keys():
                        dataset_args["jsonl_path_list"] = [meta_info["jsonl_path"]]
                    else:
                        dataset_args["jsonl_path_list"].append(meta_info["jsonl_path"])

            resume_data_status = dataset_args.pop("resume_data_status", True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args,
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr=0,
            sample_lens=list(),
            packed_position_ids=list(),
            nested_attention_masks=list(),
            split_lens=list(),
            attn_modes=list(),
            packed_text_ids=list(),
            packed_text_indexes=list(),
            packed_label_ids=list(),
            ce_loss_indexes=list(),
            ce_loss_weights=list(),
            vae_image_tensors=list(),
            packed_latent_position_ids=list(),
            vae_latent_shapes=list(),
            packed_vae_token_indexes=list(),
            packed_timesteps=list(),
            mse_loss_indexes=list(),
            packed_vit_tokens=list(),
            vit_token_seqlens=list(),
            packed_vit_position_ids=list(),
            packed_vit_token_indexes=list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status["sample_lens"]),
            sample_lens=sequence_status["sample_lens"],
            packed_text_ids=torch.tensor(sequence_status["packed_text_ids"]),
            packed_text_indexes=torch.tensor(sequence_status["packed_text_indexes"]),
            packed_position_ids=torch.tensor(sequence_status["packed_position_ids"]),
        )
        if not self.use_flex:
            data["nested_attention_masks"] = sequence_status["nested_attention_masks"]
        else:
            sequence_len = data["sequence_length"]
            pad_len = self.max_num_tokens - sequence_len
            data["split_lens"] = sequence_status["split_lens"] + [pad_len]
            data["attn_modes"] = sequence_status["attn_modes"] + ["causal"]
            data["sample_lens"] += [pad_len]

        if len(sequence_status["vae_image_tensors"]) > 0:
            image_tensors = sequence_status.pop("vae_image_tensors")
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes, strict=False))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = image_tensor

            data["padded_images"] = padded_images
            data["patchified_vae_latent_shapes"] = sequence_status["vae_latent_shapes"]
            data["packed_latent_position_ids"] = torch.cat(sequence_status["packed_latent_position_ids"], dim=0)
            data["packed_vae_token_indexes"] = torch.tensor(sequence_status["packed_vae_token_indexes"])

        if len(sequence_status["packed_vit_tokens"]) > 0:
            data["packed_vit_tokens"] = torch.cat(sequence_status["packed_vit_tokens"], dim=0)
            data["packed_vit_position_ids"] = torch.cat(sequence_status["packed_vit_position_ids"], dim=0)
            data["packed_vit_token_indexes"] = torch.tensor(sequence_status["packed_vit_token_indexes"])
            data["vit_token_seqlens"] = torch.tensor(sequence_status["vit_token_seqlens"])

        if len(sequence_status["packed_timesteps"]) > 0:
            data["packed_timesteps"] = torch.tensor(sequence_status["packed_timesteps"])
            data["mse_loss_indexes"] = torch.tensor(sequence_status["mse_loss_indexes"])

        if len(sequence_status["packed_label_ids"]) > 0:
            data["packed_label_ids"] = torch.tensor(sequence_status["packed_label_ids"])
            data["ce_loss_indexes"] = torch.tensor(sequence_status["ce_loss_indexes"])
            data["ce_loss_weights"] = torch.tensor(sequence_status["ce_loss_weights"])

        return data

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[: i + 1]) / total_weights for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            if sequence_status["curr"] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            num_tokens = sample["num_tokens"] + 2 * len(sample["sequence_plan"])
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample["data_indexes"])
                                break
                            else:
                                print(f"skip a sample with length {num_tokens}")
                                continue

            if sequence_status["curr"] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            num_tokens = sample["num_tokens"] + 2 * len(sample["sequence_plan"])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status["curr"] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                    data = self.to_tensor(sequence_status)
                    data["batch_data_indexes"] = batch_data_indexes
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample["data_indexes"])

            if sequence_status["curr"] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data["batch_data_indexes"] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample["image_tensor_list"]
        text_ids_list = sample["text_ids_list"]
        sequence_plan = sample["sequence_plan"]

        split_lens, attn_modes = list(), list()
        curr = sequence_status["curr"]
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get("split_start", True)
            if split_start:
                curr_split_len = 0

            if item["type"] == "text":
                text_ids = text_ids_list.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status["packed_text_ids"].extend(shifted_text_ids)
                sequence_status["packed_text_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
                if item["loss"] == 1:
                    sequence_status["ce_loss_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status["ce_loss_weights"].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status["packed_label_ids"].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                sequence_status["packed_text_ids"].append(self.eos_token_id)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                attn_modes.append("causal")
                sequence_status["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                image_tensor = image_tensor_list.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status["packed_vit_token_indexes"].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status["packed_vit_tokens"].append(vit_tokens)
                sequence_status["vit_token_seqlens"].append(num_img_tokens)
                sequence_status["packed_vit_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vit_patch_size,
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side,
                    )
                )

                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                attn_modes.append("full")
                sequence_status["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item["type"] == "vae_image":
                image_tensor = image_tensor_list.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                sequence_status["vae_image_tensors"].append(image_tensor)
                sequence_status["packed_latent_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vae_image_downsample,
                        max_num_patches_per_side=self.data_config.max_latent_size,
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status["vae_latent_shapes"].append((h, w))

                num_img_tokens = w * h
                sequence_status["packed_vae_token_indexes"].extend(range(curr, curr + num_img_tokens))
                if item["loss"] == 1:
                    sequence_status["mse_loss_indexes"].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float("-inf")

                sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                if split_start:
                    if item["loss"] == 1 and "frame_delta" not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status["packed_position_ids"].extend([curr_rope_id] * (num_img_tokens + 2))
                if "frame_delta" in item.keys():
                    curr_rope_id += item["frame_delta"]
                elif item["loss"] == 0:
                    curr_rope_id += 1

            if item.get("split_end", True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status["curr"] = curr
        sequence_status["sample_lens"].append(sample_lens)
        if not self.use_flex:
            sequence_status["nested_attention_masks"].append(prepare_attention_mask_per_sample(split_lens, attn_modes))
        else:
            sequence_status["split_lens"].extend(split_lens)
            sequence_status["attn_modes"].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data["batch_data_indexes"]
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, "padded_images"):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, "packed_timesteps"):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, "packed_vit_tokens"):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, "packed_label_ids"):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, "padded_images"):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, "packed_timesteps"):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, "packed_vit_tokens"):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, "packed_label_ids"):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length=self.sequence_length,
            sample_lens=self.sample_lens,
            packed_text_ids=self.packed_text_ids,
            packed_text_indexes=self.packed_text_indexes,
            packed_position_ids=self.packed_position_ids,
            batch_data_indexes=self.batch_data_indexes,
        )

        if not self.use_flex:
            data["nested_attention_masks"] = self.nested_attention_masks
        else:
            data["split_lens"] = self.split_lens
            data["attn_modes"] = self.attn_modes

        if hasattr(self, "padded_images"):
            data["padded_images"] = self.padded_images
            data["patchified_vae_latent_shapes"] = self.patchified_vae_latent_shapes
            data["packed_latent_position_ids"] = self.packed_latent_position_ids
            data["packed_vae_token_indexes"] = self.packed_vae_token_indexes

        if hasattr(self, "packed_vit_tokens"):
            data["packed_vit_tokens"] = self.packed_vit_tokens
            data["packed_vit_position_ids"] = self.packed_vit_position_ids
            data["packed_vit_token_indexes"] = self.packed_vit_token_indexes
            data["vit_token_seqlens"] = self.vit_token_seqlens

        if hasattr(self, "packed_timesteps"):
            data["packed_timesteps"] = self.packed_timesteps
            data["mse_loss_indexes"] = self.mse_loss_indexes

        if hasattr(self, "packed_label_ids"):
            data["packed_label_ids"] = self.packed_label_ids
            data["ce_loss_indexes"] = self.ce_loss_indexes
            data["ce_loss_weights"] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)

    return collate_fn


__all__ = [
    "DATASET_INFO",
    "DATASET_REGISTRY",
    "DataConfig",
    "PackedDataset",
    "SimpleCustomBatch",
    "add_special_tokens",
    "collate_wrapper",
]
