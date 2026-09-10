
import atexit
import json
import os

import torch


ENABLED = os.getenv("FFN_ABLATE", "0") == "1"
BLOCK = int(os.getenv("FFN_ABLATE_BLOCK", "64"))

RATIOS = tuple(
    float(x)
    for x in os.getenv(
        "FFN_ABLATE_KEEP",
        "0.25,0.50,0.75",
    ).split(",")
)

LAYERS = {
    int(x)
    for x in os.getenv(
        "FFN_ABLATE_LAYERS",
        "0,8,16,24,31",
    ).split(",")
    if x.strip()
}

OUTPUT = os.path.expanduser(
    os.getenv(
        "FFN_ABLATE_OUT",
        "~/Downloads/llada_ffn_ablation_m336.json",
    )
)

_records = []
_calls = {}
_block_ids = {}


def _layer_id(block):
    for name in ("layer_id", "layer_idx", "block_idx"):
        value = getattr(block, name, None)
        if isinstance(value, int):
            return value

    key = id(block)
    if key not in _block_ids:
        _block_ids[key] = len(_block_ids)

    return _block_ids[key]


def _save():
    if not _records:
        return

    parent = os.path.dirname(os.path.abspath(OUTPUT))
    os.makedirs(parent, exist_ok=True)

    temporary = OUTPUT + ".tmp"

    with open(temporary, "w") as handle:
        json.dump(
            {
                "config": {
                    "block_size": BLOCK,
                    "keep_ratios": RATIOS,
                    "layers": sorted(LAYERS),
                },
                "records": _records,
            },
            handle,
            indent=2,
        )

    os.replace(temporary, OUTPUT)


def _union_fraction(mask, group_tokens):
    values = []

    for start in range(0, mask.shape[0], group_tokens):
        group = mask[start:start + group_tokens]
        values.append(group.amax(dim=0).mean())

    return float(torch.stack(values).mean().item())


def _token_mask(energy, keep_blocks):
    indices = torch.topk(
        energy,
        k=keep_blocks,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    mask = torch.zeros_like(energy)
    mask.scatter_(1, indices, 1.0)
    return mask


def _group_mask(energy, keep_blocks, group_tokens):
    mask = torch.zeros_like(energy)

    for start in range(0, energy.shape[0], group_tokens):
        end = min(start + group_tokens, energy.shape[0])

        scores = energy[start:end].sum(dim=0)
        indices = torch.topk(
            scores,
            k=keep_blocks,
            largest=True,
            sorted=False,
        ).indices

        row = torch.zeros_like(scores)
        row.scatter_(0, indices, 1.0)

        mask[start:end] = row.unsqueeze(0).expand(
            end - start,
            -1,
        )

    return mask


def _global_mask(energy, keep_blocks):
    scores = energy.sum(dim=0)

    indices = torch.topk(
        scores,
        k=keep_blocks,
        largest=True,
        sorted=False,
    ).indices

    row = torch.zeros_like(scores)
    row.scatter_(0, indices, 1.0)

    return row.unsqueeze(0).expand(
        energy.shape[0],
        -1,
    )


def _output_error(block, activation, block_mask, reference):
    feature_mask = block_mask.repeat_interleave(
        BLOCK,
        dim=1,
    )

    feature_mask = feature_mask.reshape(
        activation.shape
    ).to(dtype=activation.dtype)

    candidate = block.ff_out(
        activation * feature_mask
    )

    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()

    difference = candidate_float - reference_float

    numerator = difference.square().sum().sqrt()
    denominator = (
        reference_float.square()
        .sum()
        .sqrt()
        .clamp_min(1e-12)
    )

    return float((numerator / denominator).item())


def probe_ffn_activation(block, activation):
    if not ENABLED:
        return activation

    layer = _layer_id(block)
    step = _calls.get(layer, 0)
    _calls[layer] = step + 1

    if layer not in LAYERS or step != 0:
        return activation

    with torch.no_grad():
        hidden = activation.shape[-1]

        if hidden % BLOCK != 0:
            raise RuntimeError(
                f"hidden={hidden} is not divisible by block={BLOCK}"
            )

        flat = activation.detach().reshape(
            -1,
            hidden,
        ).float()

        tokens = flat.shape[0]
        num_blocks = hidden // BLOCK

        energy = (
            flat.reshape(tokens, num_blocks, BLOCK)
            .square()
            .sum(dim=-1)
        )

        total_energy = energy.sum().clamp_min(1e-20)

        # 真实 dense down projection，作为误差参考。
        reference = block.ff_out(activation)

        record = {
            "layer": layer,
            "step": step,
            "tokens": tokens,
            "hidden": hidden,
            "results": [],
        }

        for ratio in RATIOS:
            keep_blocks = max(
                1,
                min(
                    num_blocks,
                    round(num_blocks * ratio),
                ),
            )

            masks = {
                "token": _token_mask(
                    energy,
                    keep_blocks,
                ),
                "group8": _group_mask(
                    energy,
                    keep_blocks,
                    8,
                ),
                "group16": _group_mask(
                    energy,
                    keep_blocks,
                    16,
                ),
                "global": _global_mask(
                    energy,
                    keep_blocks,
                ),
            }

            for strategy, mask in masks.items():
                retained = float(
                    (
                        (energy * mask).sum()
                        / total_energy
                    ).item()
                )

                output_error = _output_error(
                    block,
                    activation,
                    mask,
                    reference,
                )

                union8 = _union_fraction(mask, 8)
                union16 = _union_fraction(mask, 16)
                union_all = float(
                    mask.amax(dim=0).mean().item()
                )

                result = {
                    "keep_ratio": ratio,
                    "strategy": strategy,
                    "activation_energy_retained": retained,
                    "output_relative_l2": output_error,
                    "union_group8": union8,
                    "union_group16": union16,
                    "union_all": union_all,
                }

                record["results"].append(result)

                print(
                    "[ffn-ablate] "
                    f"L={layer:02d} "
                    f"keep={ratio:.0%} "
                    f"{strategy:7s} "
                    f"energy={retained:.2%} "
                    f"outL2={output_error:.4f} "
                    f"U8={union8:.2%} "
                    f"U16={union16:.2%}",
                    flush=True,
                )

        _records.append(record)
        _save()

        exit_layer = os.getenv(
            "FFN_ABLATE_EXIT_AFTER_LAYER"
        )

        if (
            exit_layer is not None
            and layer == int(exit_layer)
        ):
            print(
                f"[ffn-ablate] collected through layer {layer}; exiting early",
                flush=True,
            )
            raise SystemExit(0)

    return activation


def _finish():
    if _records:
        _save()
        print(
            f"[ffn-ablate] saved to {OUTPUT}",
            flush=True,
        )


atexit.register(_finish)
