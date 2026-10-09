"""Read TVR captions without assuming five annotations per video."""

import re

from utils.holmes_ar import read_captions


def split_caption_id(caption_id):
    match = re.fullmatch(r"([^#]+)#enc#(0|[1-9][0-9]*)", caption_id)
    if match is None:
        raise ValueError(f"Expected a base TVR caption ID, got {caption_id!r}")
    return match[1], int(match[2])


def build_video_context_db(caption_path):
    contexts = {}
    for caption_id, query in read_captions(caption_path).items():
        video_id, index = split_caption_id(caption_id)
        contexts.setdefault(video_id, {})[index] = query
    return contexts


def get_sibling_queries(context_db, video_id, target_idx):
    captions = context_db[video_id]
    if target_idx not in captions:
        raise KeyError((video_id, target_idx))
    return [query for index, query in captions.items() if index != target_idx]
