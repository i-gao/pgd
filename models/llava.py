from typing import List, Union

from PIL import Image
import torch
import shutil

from models.eval_model import BaseEvalModel
from utils import unwrap_model, stack_with_padding, normalize_images
from transformers.modeling_outputs import CausalLMOutputWithPast

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.realpath(__file__)) + "/../LLaVA"
)  # assume the open_flamingo repo is nested in this one

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import tokenizer_image_token


class EvalModel(BaseEvalModel):
    """LLaVA model evaluation."""

    def __init__(self, **model_args):
        assert (
            "model_path" in model_args and "model_base" in model_args
        ), "LLaVA requires model_path and model_base arguments to be specified"
        from llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria

        model_name = get_model_name_from_path(model_args["model_path"])
        model_args["lm_path"] = model_name
        super().__init__(model_args)
        (
            self.tokenizer,
            self.model,
            self.processor,
            self.context_len,
        ) = load_pretrained_model(
            model_args["checkpoint_path"],
            model_args["model_path"],
            model_args["model_base"],
            model_name,
        )
        self.conv = get_conv(model_name)
        if model_args.get("strip_filler_messages", False):
            self.conv.messages = ()
        self.stopping_criteria_class = KeywordsStoppingCriteria
        self.use_llava_prefix = True
        self.user_token = (self.conv.sep + self.conv.roles[0]).strip() + ':'
        self.agent_token = (self.conv.sep + self.conv.roles[1]).strip() + ':'
        self._check_init()

    def change_user_agent_token(self, user_token, agent_token):
        assert user_token.endswith(':') and agent_token.endswith(':')
        self.user_token = user_token
        self.agent_token = agent_token
    
    def _prepare_images(
        self, batch: Union[List[List[Image.Image]], torch.Tensor], normalize=True
    ) -> torch.Tensor:
        """Return a tensor of images of shape (B, 1, C, H, W)"""
        if torch.is_tensor(batch):
            if batch.ndim == 4:  # (B, C, H, W) -> (B, 1, C, H, W)
                batch = batch.unsqueeze(1)
            elif batch.ndim == 6:  # (B, 1, 1, C, H, W) -> (B, 1, C, H, W)
                batch = batch.squeeze(2)
            if normalize:
                batch = normalize_images(batch)
            return batch.to(self.device, dtype=self.cast_dtype, non_blocking=True)

        batch_images = None
        assert all(
            len(example) == 1 for example in batch
        ), "LLaVA only supports one image per example"
        for example in batch:
            if batch_images is None:
                batch_images = self.processor.preprocess(
                    example, do_normalize=normalize, return_tensors="pt"
                )["pixel_values"]
            else:
                batch_images = torch.cat(
                    [
                        batch_images,
                        self.processor.preprocess(
                            example, do_normalize=normalize, return_tensors="pt"
                        )["pixel_values"],
                    ]
                )
        if batch_images is not None:
            batch_images = batch_images.to(
                self.device, dtype=self.cast_dtype, non_blocking=True
            )
        return batch_images.unsqueeze(1)

    def _prepare_text(
        self,
        batch: List[str],
        padding="longest",
        truncation=True,
        max_length=2000,
        add_special_tokens=True,
        use_llava_prefix=None,
    ):
        """
        Args:
            use_llava_prefix: if not None, sets self.use_llava_prefix.
                If that's True and add_special_tokens is on, add the llava conversational prefix to the prompt
                "<s> A chat between a curious human and an artificial intelligence assistant.
                The assistant gives helpful, detailed, and polite answers to the human's questions.
                USER: {text} ASSISTANT: "
        """
        assert padding == "longest"
        if use_llava_prefix is not None:
            self.use_llava_prefix = use_llava_prefix

        # wrap image tokens
        replace_token = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
        batch = [prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token) for prompt in batch]

        # preprocess conv stuff
        if self.use_llava_prefix and add_special_tokens:
            new_batch = []
            for text in batch:
                conv = self.conv.copy()
                conv.append_message(conv.roles[0], text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # remove excess role tokens added by conv
                # last agent token 
                prompt = replace_last(prompt, self.agent_token, '')
                # duplicate user tokens
                prompt = prompt.replace(self.user_token + self.user_token, self.user_token)
                prompt = prompt.replace(self.user_token + " " + self.user_token, self.user_token)
                prompt = prompt.strip().replace("  ", " ")
                new_batch.append(prompt)
            batch = new_batch.copy()

        # tokenize
        input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            for prompt in batch
        ]
        if not add_special_tokens:
            # remove bos_token_id
            input_ids = [
                ids[1:] if ids[0] == self.tokenizer.bos_token_id else ids
                for ids in input_ids
            ]
        longest = max([len(ids) for ids in input_ids])
        input_ids = [
            torch.cat(
                [
                    torch.LongTensor(
                        [self.tokenizer.pad_token_id] * (longest - len(ids))
                    ),
                    ids,
                ]
            )
            for ids in input_ids
        ]
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        return input_ids, attention_mask

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        **decode_kwargs,
    ) -> List[str]:
        # convert to tensors and do inference
        batch_images = self._prepare_images(batch_images).squeeze(1)
        input_ids, attention_mask = self._prepare_text(batch_text)
        stop_str = (
            self.conv.sep
            if self.conv.sep_style != SeparatorStyle.TWO
            else self.conv.sep2
        )
        keywords = [stop_str]
        stopping_criteria = self.stopping_criteria_class(
            keywords, self.tokenizer, input_ids
        )

        # llava only supports one example at a time
        outputs = []
        for i in range(len(batch_text)):
            with torch.inference_mode():
                with self.autocast():
                    generation = unwrap_model(self.model).generate(
                        input_ids=input_ids[i].unsqueeze(0),
                        images=batch_images[i].unsqueeze(0),
                        attention_mask=attention_mask[i].unsqueeze(0),
                        stopping_criteria=[stopping_criteria],
                        **decode_kwargs,
                    )
                outputs.append(generation[0])

        # pad and stack outputs
        outputs = stack_with_padding(
            outputs,
            padding_value=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side,
        )

        # Extract only the new generated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [
            o[: -len(stop_str)].strip() if o.endswith(stop_str) else o.strip()
            for o in outputs
        ]
        return outputs

    def get_vqa_prompt(self, question, answer=None) -> str:
        return (
            f"Question:{question} Short answer:{answer if answer is not None else ''}"
        )

    def get_caption_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        use_cache: bool = False,
    ):
        # standard forward pass
        if past_key_values is None:
            with self.autocast():
                outputs = self.model(
                    images=vision_x,
                    input_ids=lang_x,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            outputs.logits = self._slice_to_logits(lang_x, outputs.logits)
            return outputs

        # loop to handle updating past_key_values
        logits = []
        for token_idx in range(lang_x.shape[1]):
            _lang_x = lang_x[:, token_idx].reshape((-1, 1))
            if attention_mask is not None:
                _attention_mask = attention_mask[:, token_idx].reshape((-1, 1))
            else:
                _attention_mask = None

            with self.autocast():
                outputs = self.model(
                    images=vision_x,
                    input_ids=_lang_x,
                    attention_mask=_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits.append(outputs.logits)

        logits = torch.cat(logits, dim=1)
        logits = self._slice_to_logits(lang_x, logits)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def _slice_to_logits(self, input_ids, logits):
        """
        Given a batch of input_ids and logits, remove the logits corresponding to the image tokens
        Each <image> token in input_ids gets internally replaced with 256 image tokens
        """
        # Loop through the batch and examples
        batch_logits = []
        for i in range(input_ids.shape[0]):
            sequence_logits = []
            logits_j = 0
            for j in range(input_ids.shape[1]):
                if input_ids[i, j] != IMAGE_TOKEN_INDEX:
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += 1
                else:
                    # append the logit for the first image token, then skip over the rest
                    # note: the model actually learns to predict <im_patch>, not <image>
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += 256
            sequence_logits = torch.stack(
                sequence_logits, dim=0
            )  # (seq_len, vocab_size)
            batch_logits.append(sequence_logits)

        batch_logits = torch.stack(
            batch_logits, dim=0
        )  # (batch_size, seq_len, vocab_size)
        # The final logits shape should be the same as the original input_ids shape
        assert batch_logits.shape[:2] == input_ids.shape
        return batch_logits

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        raise NotImplementedError(
            "LLaVA classification-based evaluation not implemented"
        )


def get_conv(model_name):
    if "v0" in model_name.lower():
        template_name = "v0"
    elif "llava" in model_name.lower():
        if "v1" in model_name.lower():
            template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt_multimodal"
        else:
            template_name = "multimodal"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "koala" in model_name:  # Hardcode the condition
        template_name = "bair_v1"
    elif "v1" in model_name:  # vicuna v1_1/v1_2
        template_name = "vicuna_v1_1"
    else:
        template_name = "v1"
    return conv_templates[template_name].copy()


def load_pretrained_model(checkpoint_path, model_path, model_base, device_map="auto"):
    from llava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
    from transformers import AutoTokenizer, AutoConfig

    # initialize model
    if "mpt" in model_path.lower():
        if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
            shutil.copyfile(
                os.path.join(model_base, "configuration_mpt.py"),
                os.path.join(model_path, "configuration_mpt.py"),
            )
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
        cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LlavaMPTForCausalLM.from_pretrained(model_base, config=cfg_pretrained)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_base, config=cfg_pretrained)

    # load checkpoint
    mm_projector_weights = torch.load(
        os.path.join(checkpoint_path, "mm_projector.bin"), map_location="cpu"
    )
    model.load_state_dict(mm_projector_weights, strict=False)

    # configure image start / end tokens
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    # get image processor
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail
