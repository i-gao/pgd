import torch
from PIL import Image
import os
import torch.nn as nn
from contextlib import suppress

IMAGENET_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def check_image_unnormalized(x):
    """Heuristically assert that the image is in [0, 1] space"""
    return x.max() <= 1 and x.min() >= 0


def stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)
        if len(tensor.size()) == 1:
            padding = torch.full(
                (max_tokens - num_tokens,),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            padding = torch.full(
                (max_tokens - num_tokens, tensor.size(1)),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        padded_tensor = (
            torch.cat((tensor, padding), dim=0)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)


def get_prompt_target_ids(
    model,
    prompts: [str],
    targets: [str],
    padding_side="left",
    **tokenizer_kwargs,
):
    assert len(prompts) == len(targets)
    assert all(
        [p.strip() == p for p in prompts]
    ), "Prompts should not end in trailing whitespaces!"
    assert model.tokenizer.padding_side == padding_side

    print("Before tokenization:", prompts, targets, "\n")

    # first, get the combined prompt + target input_ids
    combined = [prompts[i] + " " + targets[i] for i in range(len(prompts))]
    combined_input_ids, combined_attention_mask = model._prepare_text(
        combined, max_length=10000, **tokenizer_kwargs
    )

    # then, get the prompt input_ids
    prompt_input_ids, prompt_attention_mask = model._prepare_text(
        prompts, max_length=10000, **tokenizer_kwargs
    )

    # splice out the target input_ids
    target_lens = combined_attention_mask.sum(-1) - prompt_attention_mask.sum(-1)
    assert torch.all(
        target_lens > 0
    ), "Combined and prompt input_ids were tokenized to the same length! Maybe add a space before the target."

    target_input_ids = []
    for i in range(len(prompt_input_ids)):
        target_input_ids.append(combined_input_ids[i, -target_lens[i] :])
    target_input_ids = stack_with_padding(
        target_input_ids,
        padding_value=model.tokenizer.pad_token_id,
        padding_side=padding_side,
    ).to(dtype=combined_input_ids.dtype, device=combined_input_ids.device)
    target_attention_mask = (target_input_ids != model.tokenizer.pad_token_id).to(
        dtype=combined_attention_mask.dtype, device=combined_attention_mask.device
    )

    # debugging: batch decode and print
    def _print_me(ids, str=""):
        ids_copy = ids.clone()
        ids_copy[ids_copy < 0] = 0
        print(str, model.tokenizer.batch_decode(ids_copy))

    _print_me(combined_input_ids, str="Combined tokens:")
    _print_me(prompt_input_ids, str="Prompt tokens:")
    _print_me(target_input_ids, str="Target tokens:")

    return {
        "combined": (combined_input_ids, combined_attention_mask),
        "prompt": (prompt_input_ids, prompt_attention_mask),
        "target": (target_input_ids, target_attention_mask),
    }


def save_images(images, normalized=False, dirname=None, name="testing", wandb=None):
    """Save a 5- or 6-dim tensor of images to a directory"""
    if torch.is_tensor(images):
        images = tensor_to_image(images, normalized=normalized)

    assert isinstance(images, list)
    if len(images):
        assert isinstance(images[0], list)

    to_save = []
    for batch_ix, sequence in enumerate(images):
        for seq_ix, img in enumerate(sequence):
            if dirname is not None:
                img.save(f"{dirname}/{name}_{batch_ix}_{seq_ix}.jpg")
            if wandb is not None:
                to_save.append(wandb.Image(img))
    if wandb is not None:
        wandb.log({f"{name}_images": to_save})


def random_seed(seed=42, rank=0):
    """Set the random seed"""
    torch.manual_seed(seed + rank)


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model


def tensor_to_image(
    x, normalized=False, image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD
):
    """
    Given a tensor of shape (B, T, C, H, W) or (B, T, 1, C, H, W), return a list of lists of PIL images
    Notes: calling this function (and re-processing later) is pretty lossy. Avoid calling this except for saving.
    """
    assert image_mean.ndim == 3 and image_std.ndim == 3
    if x.ndim == 6:
        x = x.squeeze(2)
    assert x.ndim == 5
    assert not (
        not normalized and not check_image_unnormalized(x)
    ), "Flag says that image is not normalized (in [0, 1] space), but it is (not in [0, 1] space)!"
    if normalized:
        x = unnormalize_images(x, image_mean=image_mean, image_std=image_std)

    images = []
    for batch_ix in range(x.shape[0]):
        images.append([])
        for seq_ix in range(x.shape[1]):
            img = 255 * x[batch_ix, seq_ix]
            img = img.to(dtype=torch.uint8)
            img = torch.permute(img, (1, 2, 0))
            img = img.cpu().numpy()
            img = Image.fromarray(img, "RGB")
            images[-1].append(img)
    return images


def unnormalize_images(x, image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD):
    """
    Unnormalize a batch of images, where x is a tensor of shape (B, T, C, H, W) or (B, T, 1, C, H, W)
    """
    has_frame_dim = x.ndim == 6
    if has_frame_dim:
        x = x.squeeze(2)
    assert x.ndim == 5
    x = x * image_std[None, None, :, :, :].to(x.device) + image_mean[
        None, None, :, :, :
    ].to(x.device)
    if has_frame_dim:
        x = x.unsqueeze(2)
    return x


def normalize_images(x, image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD):
    """
    Normalize a batch of images, where x is a tensor of shape (B, T, C, H, W) or (B, T, 1, C, H, W)
    """
    has_frame_dim = x.ndim == 6
    if has_frame_dim:
        x = x.squeeze(2)
    assert x.ndim == 5
    x = (x - image_mean[None, None, :, :, :].to(x.device)) / image_std[
        None, None, :, :, :
    ].to(x.device)
    if has_frame_dim:
        x = x.unsqueeze(2)
    return x
