from typing import List, Union

from PIL import Image
import torch
from einops import repeat

from models.eval_model import BaseEvalModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import unwrap_model, normalize_images
from open_clip.transform import image_transform

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.realpath(__file__)) + "/../open_flamingo"
)  # assume the open_flamingo repo is nested in this one
from open_flamingo.src.factory import create_model_and_transforms


class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation."""

    def __init__(self, **model_args):
        assert (
            "vision_encoder_path" in model_args
            and "lm_path" in model_args
            and "checkpoint_path" in model_args
            and "lm_tokenizer_path" in model_args
            and "cross_attn_every_n_layers" in model_args
            and "vision_encoder_pretrained" in model_args
        ), "OpenFlamingo requires vision_encoder_path, lm_path, device, checkpoint_path, lm_tokenizer_path, cross_attn_every_n_layers, vision_encoder_pretrained arguments to be specified"
        super().__init__(model_args)
        (
            self.model,
            self.image_processor,
            self.tokenizer,
        ) = create_model_and_transforms(
            model_args["vision_encoder_path"],
            model_args["vision_encoder_pretrained"],
            model_args["lm_path"],
            model_args["lm_tokenizer_path"],
            cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
        )
        checkpoint = torch.load(model_args["checkpoint_path"], map_location="cpu")
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        self.unnormalized_image_processor = image_transform(
            image_size=224, is_train=False, mean=(0, 0, 0), std=(1, 1, 1)
        )
        self.user_token = "[USER]:"
        self.agent_token = "[AGENT]:"
        self._check_init()

    def change_user_agent_token(self, user_token, agent_token):
        assert user_token.endswith(':') and agent_token.endswith(':')
        self.user_token = user_token
        self.agent_token = agent_token

    def _prepare_images(
        self, batch: Union[List[List[Image.Image]], torch.Tensor], normalize=True
    ) -> torch.Tensor:
        """Return a tensor of images of shape (B, T, F, C, H, W)"""
        if torch.is_tensor(batch):
            if batch.ndim == 4: # (B, C, H, W) -> (B, 1, 1, C, H, W)
                batch = batch.unsqueeze(1).unsqueeze(1)
            elif batch.ndim == 5: # (B, T, C, H, W) -> (B, T, 1, C, H, W)
                batch = batch.unsqueeze(2)
            if normalize:
                batch = normalize_images(batch)
            return batch.to(
                self.device, dtype=self.cast_dtype, non_blocking=True
            )

        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = (
                    self.image_processor(image)
                    if normalize
                    else self.unnormalized_image_processor(image)
                )
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(
                self.device, dtype=self.cast_dtype, non_blocking=True
            )
        return batch_images

    def _prepare_text(
        self,
        batch: List[str],
        padding="longest",
        truncation=True,
        max_length=2000,
        add_special_tokens=True,
    ):
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        return input_ids, attention_mask.bool()

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        **decode_kwargs,
    ) -> List[str]:
        batch_images = self._prepare_images(batch_images)  # (B, T, 1, C, H, W)
        input_ids, attention_mask = self._prepare_text(batch_text)

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    vision_x=batch_images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    **decode_kwargs,
                )

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
        """
        batch_images = self._prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        # Cache the context
        if use_cache:
            # reserve the last token in the context for the main forward pass
            self.cache_media(
                input_ids=ctx_input_ids,
                vision_x=batch_images,
            )
            with torch.inference_mode():
                precomputed = self.__call__(
                    vision_x=None,
                    lang_x=ctx_input_ids,
                    attention_mask=ctx_attention_mask,
                    clear_conditioned_layers=False,
                    use_cache=True,
                )
            precomputed_logits = precomputed.logits
            precomputed_pkvs = precomputed.past_key_values
        else:
            precomputed_pkvs = None

        # Loop through class names and get log-likelihoods
        # Note: if all classnames are one token, this code is redundant, since we could
        # get all logits after one pass. However, if there are multi-token classnames,
        # we need to loop through each classname separately.
        overall_probs = []
        for class_name in all_class_names:
            # Tokenize only the class name
            classname_tokens = self.tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.device)
            assert classname_tokens.ndim == 2
            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )
            num_tokens_in_classname = classname_tokens.shape[1]

            # Concatenate the class name tokens
            if not use_cache:
                _lang_x = torch.cat([ctx_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )
                _vision_x = batch_images
            else:
                _lang_x = classname_tokens
                _attention_mask = None
                _vision_x = None

            # Call forward to get the logits
            with torch.inference_mode():
                outputs = self.__call__(
                    vision_x=_vision_x,
                    lang_x=_lang_x,
                    attention_mask=_attention_mask,
                    clear_conditioned_layers=(not use_cache),
                    past_key_values=precomputed_pkvs,
                )

            # Get the logits of the classname
            # logits shape is either (B, num_tokens_in_classname, vocab_len) with use_cache
            # or (B, len(_lang_x), vocab_len) without use_cache
            # remember that the logits at index t on dim 1 correspond to predictions for the t+1st token
            logits = outputs.logits
            if use_cache:
                logits = torch.cat([precomputed_logits, logits], dim=1)

            logprobs = torch.log_softmax(logits, dim=-1)
            gen_probs = logprobs[
                :, -num_tokens_in_classname - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
            gen_probs = torch.gather(
                gen_probs, 2, classname_tokens[:, :, None]
            ).squeeze(-1)

            # Aggregate over tokens in the classname
            if normalize_length:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
            overall_probs.append(class_prob)  # (B, 1)

        self.uncache_media()
        overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
        return overall_probs

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
        use_cache: bool = False,
    ):
        if vision_x.ndim == 5:  # no frame dimension
            vision_x = vision_x.unsqueeze(2)

        # standard forward pass
        if past_key_values is None:
            with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    clear_conditioned_layers=clear_conditioned_layers,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
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
                    vision_x=vision_x,
                    lang_x=_lang_x,
                    attention_mask=_attention_mask,
                    clear_conditioned_layers=False,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits.append(outputs.logits)

        logits = torch.cat(logits, dim=1)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def get_vqa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_imagenet_prompt(self, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

    def get_hateful_memes_prompt(self, text, label=None) -> str:
        return f"<image>is an image with: '{text}' written on it. Is it hateful? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
