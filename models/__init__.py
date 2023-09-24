from .eval_model import BaseEvalModel

def load_openflamingo_model(
    checkpoint_path: str,
    clip_vision_encoder_path: str = "ViT-L-14",
    clip_vision_encoder_pretrained: str = "openai",
    lang_encoder_path: str = "lmsys/vicuna-7b-v1.3",
    tokenizer_path: str = "lmsys/vicuna-7b-v1.3",
    cross_attn_every_n_layers: int = 4,
    precision: str = "fp32",
    device: int = 0,
):
    from .open_flamingo import EvalModel

    return EvalModel(
        vision_encoder_path=clip_vision_encoder_path,
        vision_encoder_pretrained=clip_vision_encoder_pretrained,
        lm_path=lang_encoder_path,
        lm_tokenizer_path=tokenizer_path,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        checkpoint_path=checkpoint_path,
        precision=precision,
        device=device,
    )


def load_blip_model(
    lm_path: str = "Salesforce/blip2-opt-2.7b",
    processor_path: str = "Salesforce/blip2-opt-2.7b",
    precision: str = "fp32",
    device: int = 0,
):
    from .blip import EvalModel

    return EvalModel(
        lm_path=lm_path,
        processor_path=processor_path,
        precision=precision,
        device=device,
    )


def load_idefics_model(
    lm_path: str = "HuggingFaceM4/idefics-9b",
    processor_path: str = "HuggingFaceM4/idefics-9b",
    precision: str = "fp32",
    device: int = 0,
):
    from .idefics import EvalModel

    return EvalModel(
        lm_path=lm_path,
        processor_path=processor_path,
        precision=precision,
        device=device,
    )

def load_llava_model(
    checkpoint_path: str = "/sailhome/irena/juicehome/openflamingo_checkpoints/llava-7B",
    model_path: str = "liuhaotian/llava-pretrain-vicuna-7b-v1.3",
    model_base: str = "lmsys/vicuna-7b-v1.3",
    precision: str = "fp32",
    device: int = 0,
):
    from .llava import EvalModel

    return EvalModel(
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        model_base=model_base,
        precision=precision,
        device=device,
        strip_filler_messages=True,
    )


def load_clip(
    clip_vision_encoder_path: str = "ViT-L-14",
    clip_vision_encoder_pretrained: str = "openai",
):
    """Loads the CLIP model"""
    import open_clip

    clip, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    tokenize_fn = open_clip.tokenize
    return clip, image_processor, tokenize_fn
