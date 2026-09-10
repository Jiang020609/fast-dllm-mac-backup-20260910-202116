
import atexit
import json
import os

import torch


ENABLED = os.getenv("FFN_SPARSE_PROBE", "0") == "1"
BLOCK = int(os.getenv("FFN_SPARSE_BLOCK", "64"))
MAX_STEPS = int(os.getenv("FFN_SPARSE_MAX_STEPS", "3"))

KEEP_RATIOS = tuple(
    float(x)
    for x in os.getenv("FFN_SPARSE_KEEP", "0.25,0.50").split(",")
)

LAYERS = {
    int(x)
    for x in os.getenv(
        "FFN_SPARSE_LAYERS",
        "0,8,16,24,31",
    ).split(",")
    if x.strip()
}

OUTPUT = os.path.expanduser(
    os.getenv(
        "FFN_SPARSE_OUT",
        "~/Downloads/llada_ffn_sparsity.json",
    )
)

_records = []
_block_ids = {}
_layer_calls = {}
_previous_shared = {}
_failed = False


def _layer_id(block):
    for name in ("layer_id", "layer_idx", "block_idx"):
        value = getattr(block, name, None)
        if isinstance(value, int):
            return value

    key = id(block)
    if key not in _block_ids:
        _block_ids[key] = len(_block_ids)
    return _block_ids[key]


def _group_union_fraction(mask, group_tokens):
    values = []

    for start in range(0, mask.shape[0], group_tokens):
        group = mask[start:start + group_tokens]
        union = group.amax(dim=0)
        values.append(union.mean())

    return float(torch.stack(values).mean().item())


def _save():
    parent = os.path.dirname(os.path.abspath(OUTPUT))
    os.makedirs(parent, exist_ok=True)

    payload = {
        "config": {
            "block_size": BLOCK,
            "keep_ratios": KEEP_RATIOS,
            "layers": sorted(LAYERS),
            "max_steps_per_layer": MAX_STEPS,
        },
        "records": _records,
    }

    temporary = OUTPUT + ".tmp"

    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)

    os.replace(temporary, OUTPUT)


def probe_ffn_activation(block, activation):
    global _failed

    if not ENABLED or _failed:
        return activation

    layer = _layer_id(block)
    step = _layer_calls.get(layer, 0)
    _layer_calls[layer] = step + 1

    if layer not in LAYERS or step >= MAX_STEPS:
        return activation

    try:
        with torch.no_grad():
            hidden = activation.shape[-1]

            if hidden % BLOCK != 0:
                raise RuntimeError(
                    f"hidden={hidden} is not divisible by block={BLOCK}"
                )

            flat = activation.detach().reshape(-1, hidden).float()
            tokens = flat.shape[0]
            num_blocks = hidden // BLOCK

            abs_flat = flat.abs()
            rms = flat.square().mean(dim=1, keepdim=True).sqrt()
            rms = rms.clamp_min(1e-12)

            exact_zero = float((abs_flat == 0).float().mean().item())
            near_001 = float((abs_flat <= 0.01 * rms).float().mean().item())
            near_005 = float((abs_flat <= 0.05 * rms).float().mean().item())
            near_010 = float((abs_flat <= 0.10 * rms).float().mean().item())

            block_energy = (
                flat.reshape(tokens, num_blocks, BLOCK)
                .square()
                .sum(dim=-1)
            )

            token_total = block_energy.sum(dim=1).clamp_min(1e-20)
            shared_scores = block_energy.sum(dim=0)
            shared_total = shared_scores.sum().clamp_min(1e-20)

            record = {
                "layer": layer,
                "step": step,
                "shape": list(activation.shape),
                "tokens": tokens,
                "hidden": hidden,
                "exact_zero_fraction": exact_zero,
                "near_0.01_rms_fraction": near_001,
                "near_0.05_rms_fraction": near_005,
                "near_0.10_rms_fraction": near_010,
                "keep": {},
            }

            for keep_ratio in KEEP_RATIOS:
                keep_blocks = max(
                    1,
                    min(
                        num_blocks,
                        round(num_blocks * keep_ratio),
                    ),
                )

                token_values, token_indices = torch.topk(
                    block_energy,
                    k=keep_blocks,
                    dim=1,
                    largest=True,
                    sorted=False,
                )

                token_energy_retained = float(
                    (
                        token_values.sum(dim=1) / token_total
                    ).mean().item()
                )

                token_mask = torch.zeros_like(block_energy)
                token_mask.scatter_(1, token_indices, 1.0)

                union_8 = _group_union_fraction(token_mask, 8)
                union_16 = _group_union_fraction(token_mask, 16)
                union_32 = _group_union_fraction(token_mask, 32)
                union_all = float(
                    token_mask.amax(dim=0).mean().item()
                )

                shared_values, shared_indices = torch.topk(
                    shared_scores,
                    k=keep_blocks,
                    largest=True,
                    sorted=False,
                )

                shared_energy_retained = float(
                    (shared_values.sum() / shared_total).item()
                )

                current_shared = set(
                    int(x)
                    for x in shared_indices.detach().cpu().tolist()
                )

                temporal_key = (layer, keep_ratio)
                previous_shared = _previous_shared.get(temporal_key)

                if previous_shared is None:
                    temporal_jaccard = None
                else:
                    intersection = len(
                        current_shared.intersection(previous_shared)
                    )
                    union = len(
                        current_shared.union(previous_shared)
                    )
                    temporal_jaccard = (
                        intersection / union if union else 1.0
                    )

                _previous_shared[temporal_key] = current_shared

                label = f"{keep_ratio:.2f}"

                record["keep"][label] = {
                    "keep_blocks": keep_blocks,
                    "token_energy_retained": token_energy_retained,
                    "shared_energy_retained": shared_energy_retained,
                    "token_mask_union_group_8": union_8,
                    "token_mask_union_group_16": union_16,
                    "token_mask_union_group_32": union_32,
                    "token_mask_union_all": union_all,
                    "shared_mask_temporal_jaccard": temporal_jaccard,
                }

            _records.append(record)
            _save()

            report = record["keep"].get("0.50")
            if report is None:
                report = next(iter(record["keep"].values()))

            temporal = report["shared_mask_temporal_jaccard"]
            temporal_text = (
                "NA" if temporal is None else f"{temporal:.3f}"
            )

            print(
                "[ffn-sparse] "
                f"L={layer:02d} step={step} M={tokens} "
                f"zero={exact_zero:.3%} "
                f"near0.1RMS={near_010:.3%} "
                f"tokenE={report['token_energy_retained']:.3%} "
                f"sharedE={report['shared_energy_retained']:.3%} "
                f"union16={report['token_mask_union_group_16']:.3%} "
                f"unionAll={report['token_mask_union_all']:.3%} "
                f"temporalJ={temporal_text}",
                flush=True,
            )

    except Exception as error:
        _failed = True
        print(
            f"[ffn-sparse] probe disabled after error: {error}",
            flush=True,
        )

    return activation


def _finish():
    if _records:
        _save()
        print(
            f"[ffn-sparse] saved {len(_records)} records to {OUTPUT}",
            flush=True,
        )


atexit.register(_finish)
