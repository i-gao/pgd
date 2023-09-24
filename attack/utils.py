import torch

def check_first_token(
    model,
    x,
    prompts,
    targets
):
    from utils import get_prompt_target_ids, normalize_images
    embs = get_prompt_target_ids(model, prompts, targets)

    # first, get the predictions for the first target token position when using just the prompt
    with torch.no_grad():
        input_ids, attention_mask = embs['prompt']
        logits = model(
            vision_x=normalize_images(x).to(model.device, dtype=model.cast_dtype),
            lang_x=input_ids,
            attention_mask=attention_mask,
        ).logits
        prompt_predicted = logits[:, -1, :].argmax(-1)
    
    # next, get the predictions for the first target token position when using the prompt + target
    with torch.no_grad():
        input_ids, attention_mask = embs['combined']
        logits = model(
            vision_x=normalize_images(x).to(model.device, dtype=model.cast_dtype),
            lang_x=input_ids,
            attention_mask=attention_mask,
        ).logits
        combined_predicted = []
        target_lens = embs['target'][1].sum(-1)
        for i in range(logits.shape[0]):
            combined_predicted.append(
                logits[i, -target_lens[i]-1, :].argmax(-1)
            )
        combined_predicted = torch.stack(combined_predicted)
    
    # finally, check if the predictions match
    print("prompt predicted: ", embs['prompt'], prompt_predicted)
    print("combined predicted: ", embs['combined'], combined_predicted)
    return torch.all(
        prompt_predicted == combined_predicted
    ).item()


def compute_loss(
    criterion,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    num_target_tokens: torch.Tensor,
):
    """
    Given logits, compute the loss wrt the last num_target_tokens tokens
    Args:
        criterion: callable, e.g. torch.nn.CrossEntropyLoss
        logits: torch.Tensor of shape (B, L, V)
        input_ids: torch.Tensor of shape (B, L)
        num_target_tokens: torch.Tensor of shape (B)
            values are <= L
    """
    labels = input_ids.clone().detach().to(logits.device)
    for i in range(input_ids.shape[0]):
        labels[i, : -num_target_tokens[i]] = -100

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(
        shift_logits.view(-1, logits.shape[-1]),
        shift_labels.view(-1),
    )
    return loss


def check_greedy_output_match(
    logits: torch.Tensor,
    target_input_ids: torch.Tensor,
    target_attention_mask: torch.Tensor,
):
    """
    Checks if the end of the greedy output of the model exactly matches the target
    Args:
        logits: torch.Tensor of shape (B, L, V)
        target_input_ids: torch.Tensor of shape (B, L_tgt)
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    target_token_lens = target_attention_mask.sum(-1)

    # Get the greedy output
    success_mask = []
    for i in range(target_input_ids.shape[0]):
        prediction = shift_logits[i, -target_token_lens[i] :, :].argmax(-1).detach()
        label = target_input_ids[i][target_attention_mask[i].bool()]
        success_mask.append(
            torch.all(
                prediction == label
            ).item()
        )
    success_mask = torch.BoolTensor(success_mask)
    return success_mask
