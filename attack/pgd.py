import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from utils import normalize_images, get_prompt_target_ids, check_image_unnormalized
from attack.utils import compute_loss, check_greedy_output_match, check_first_token
from models import BaseEvalModel


def batch_targeted_pgd(
    model: BaseEvalModel,
    attack_class,
    prompt_images: [[Image.Image]],
    attack_mask: [[bool]],
    prompts: [str],
    targets: [str],
    epsilon=0.01,
    lr=0.01,
    num_steps=100,
    early_stop=True,
):
    """
    Runs targeted projected gradient descent on a batch of (images, prompts, targets).
    Args:
        model: open_flamingo.eval.models.EvalModel with tokenizer, _prepare_images, _prepare_text
        attack_class: subclass of Attack from attack.steps
        prompt_images: list of lists of PIL.Image, or a 5-dim tensor
            shape (B, T_img) or (B, T_img, 3, H, W)
        attack_mask: list of lists of bool
            shape (B, T_img)
            represents whether to attack the image at that timestep; 1 is attack, 0 is don't attack
        prompts: list of strings
            shape (B,)
        targets: list of strings
            shape (B,)
        epsilon: float
            maximum perturbation size
        lr: float
            step size
        num_steps: int
            number of steps to run
        early_stop: bool
            whether to stop early if all images are successful, and whether to stop attacking
            an example in the batch if it is successful
    """
    assert hasattr(model, "tokenizer")
    assert hasattr(model, "_prepare_images")
    assert hasattr(model, "_prepare_text")

    # prepare input images, prompts, and targets
    x = (
        prompt_images
        if torch.is_tensor(prompt_images)
        else model._prepare_images(prompt_images, normalize=False)
    )
    assert check_image_unnormalized(x), "Image not in [0, 1] space!"
    assert x.ndim >= 5  # (B, T_img, 3, H, W)
    embs = get_prompt_target_ids(model, prompts, targets)
    target_input_ids, target_attention_mask = embs["target"]
    input_ids, attention_mask = embs["combined"]
    attack_mask = torch.tensor(attack_mask).to(model.device)

    # check_first_token(model, x, prompts, targets)

    # define the attack + loss function
    attack = attack_class(x, epsilon, lr)
    criterion = nn.CrossEntropyLoss()
    m = -1  # we want to do gradient descent (targeted attack)

    iterator = tqdm(range(num_steps + 1))
    for it in iterator:
        x = x.clone().detach().requires_grad_(True)

        # Calculate the loss (comparing the model's output to the target token)
        logits = model(
            vision_x=normalize_images(x).to(model.device, dtype=model.cast_dtype),
            lang_x=input_ids,
            attention_mask=attention_mask,
        ).logits

        loss = compute_loss(criterion, logits, input_ids, target_attention_mask.sum(-1))

        # Check if the model's greedy output is the target
        with torch.no_grad():
            success_mask = check_greedy_output_match(
                logits,
                target_input_ids,
                target_attention_mask,
            )
        iterator.set_description(
            f"Loss: {loss.item()}; num success: {success_mask.sum()}"
        )
        if early_stop and torch.all(success_mask):
            print("All succeeded!")
            # check_first_token(model, x, prompts, targets)
            break

        if it == num_steps:
            break

        # Update adversarial images
        (grad,) = torch.autograd.grad(m * loss, [x])
        if early_stop:
            grad[success_mask] = 0
        grad[~attack_mask] = 0
        with torch.no_grad():
            x = attack.step(x, grad)
            x = attack.project(x)
            assert check_image_unnormalized(x), "Image not in [0, 1] space!"

    assert check_image_unnormalized(x), "Image not in [0, 1] space!"
    return x.detach(), success_mask
