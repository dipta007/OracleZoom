import torch
from torch import nn
from lora.lora_layers import LoraInjectedLinear, LoraInjectedConv2d

def _find_modules(model, ancestor_class=None, search_class=[nn.Linear], exclude_children_of=[LoraInjectedLinear]):
    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            # if 'norm1_context' in fullname:
                if any([isinstance(module, _class) for _class in search_class]):
                    *path, name = fullname.split(".")
                    parent = ancestor
                    while path:
                        parent = parent.get_submodule(path.pop(0))
                    if exclude_children_of and any(
                        [isinstance(parent, _class) for _class in exclude_children_of]
                    ):
                        continue
                    yield parent, name, module

def extract_lora_ups_down(model, target_replace_module={'AdaLayerNormZero'}):   # Attention for kv_lora

    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        loras.append((_child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras

def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module={'AdaLayerNormZero'},     # Attention for kv_lora
    save_half:bool=False
):
    weights = []
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        dtype = torch.float16 if save_half else torch.float32
        weights.append(_up.weight.to("cpu").to(dtype))
        weights.append(_down.weight.to("cpu").to(dtype))

    torch.save(weights, path)