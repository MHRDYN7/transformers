# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Dnv2 checkpoints trained with the DINO method."""

from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision import transforms
import argparse
from transformers import Dnv2Config, Dnv2ForImageClassification, Dnv2Model, BitImageProcessor
from transformers.utils import logging
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.blocks.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.blocks.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"encoder.blocks.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"encoder.blocks.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.blocks.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.blocks.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.blocks.{i}.mlp.fc1.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.blocks.{i}.mlp.fc1.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.blocks.{i}.mlp.fc2.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.blocks.{i}.mlp.fc2.bias"))
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.blocks.{i}.layerscale1.gmma"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.blocks.{i}.layerscale2.gmma"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "embeddings.cls_token"),
            ("pos_embed", "embeddings.position_embeddings"),
            ("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"),
            ("mask_token", "embeddings.mask_token")

        ]
    )

    if base_model:
        # layernorm + pooler
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )

        # if just the base model, we should remove "dnv2" from all keys that start with "dnv2"
        #rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("dnv2") else pair for pair in rename_keys]
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("norm.weight", "dnv2.layernorm.weight"),
                ("norm.bias", "dnv2.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "dnv2."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.blocks.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.blocks.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.blocks.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.blocks.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.blocks.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.blocks.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dnv2_checkpoint(model_name=None, pytorch_dump_folder_path=None, base_model=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Dnv2 structure.
    """

    # define default Dnv2 configuration
    
    config = Dnv2Config()
    if model_name == "dinov2_vits14":
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    elif model_name == "dinov2_vitl14":
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 12
        config.num_attention_heads = 24
    elif model_name == "dinov2_vitg14":
        config.hidden_size = 1536
        config.intermediate_size = 6144
        config.num_hidden_layers = 40
        config.num_attention_heads = 24


    #model_name = 'dinov2_vitb14'
    # load original model from torch hub
    original_model = torch.hub.load("facebookresearch/dinov2", model_name)    # ? step1
    original_model.eval()                                                     # ? step2

    # load state_dict of original model, remove and rename some keys

    #original_model = torch.load("src/transformers/models/dnv2/model.pt")

    state_dict = original_model.state_dict()                                  # ? step3

    if base_model:
        remove_classification_head_(state_dict)                               # ? step4
    rename_keys = create_rename_keys(config, base_model=base_model)           # ? step5
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)                                     # ? step6
    read_in_q_k_v(state_dict, config, base_model)                             # ? step7

    # load HuggingFace model
    if base_model:
        model = Dnv2Model(config, add_pooling_layer=False).eval()
    else:
        model = Dnv2ForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by ViTImageProcessor
    # Preprocess Image
    transformations = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])

    pixel_values = transformations(prepare_img()).unsqueeze(0)

    processor = BitImageProcessor(
        size={"shortest_edge": 256},
        resample=PILImageResampling.BICUBIC,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
    )
    new_pixel_values = processor(prepare_img(), return_tensors="pt").pixel_values
    
    assert torch.allclose(pixel_values, new_pixel_values)
    
    #outputs = model.forward_features(pixel_values)




    #image_processor = ViTImageProcessor()
    #encoding = image_processor(images=prepare_img(), return_tensors="pt")
    #pixel_values = encoding["pixel_values"]

    outputs = model(new_pixel_values)

    if base_model:
        final_hidden_state_cls_token = original_model(pixel_values)
        ##print(final_hidden_state_cls_token.unsqueeze[0,:3,:3])
        print(final_hidden_state_cls_token.unsqueeze(0)[0, :3, :3])
        print(outputs.last_hidden_state[0, :3, :3])
        assert torch.allclose(final_hidden_state_cls_token, outputs.last_hidden_state[:, 0, :], atol=1e-5)
        print("it works")
    else:
        logits = original_model(pixel_values)
        assert logits.shape == outputs.logits.shape
        assert torch.allclose(logits, outputs.logits, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        #image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_name_to_hf_nmae = {"dinov2_vits14":"dnv2-small",
                                 "dinov2_vitb14":"dnv2-base", 
                                 "dinov2_vitl14":"dnv2-large", 
                                 "dinov2_vitg14":"dnv2-giant"}
        model_name = model_name_to_hf_nmae[model_name]
        model.push_to_hub(f"MHRDYN7/{model_name}", use_temp_dir=True)
        #processor.push_to_hub(f"mhrdyn7/{model_name}", use_temp_dir=True)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default='dinov2_vitb14',
        type=str,
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="Name of the model trained with Dnv2 you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--base_model",
        default=True,
        action="store_true",
        help="Whether to only convert the base model (no projection head weights).",
    )
    parser.add_argument(
        "--push_to_hub",
        default=False,
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub.",
    )

    parser.set_defaults(base_model=True)
    args = parser.parse_args()
    
    convert_dnv2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.base_model, args.push_to_hub)
