"""Dry-run MXNet pretrained-weight conversion and SSHR/HST loading."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net
from network.resnet38d import convert_mxnet_to_torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--rectifier", default="hst", choices=["hfrm", "hst"])
    parser.add_argument(
        "--hst_variant", default="a1", choices=["a1", "a2", "a3"]
    )
    args = parser.parse_args()

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    converted = convert_mxnet_to_torch(str(weights_path))
    model_kwargs = {"rectifier_type": args.rectifier}
    if args.rectifier == "hst":
        model_kwargs["hst_config"] = {"variant": args.hst_variant}
    model = Net(n_class=4, **model_kwargs)
    incompatible = model.load_state_dict(converted, strict=False)

    rectifier_prefixes = (
        "hfrm_56.",
        "hfrm_28_1.",
        "hfrm_28_2.",
        "hst_rectifier.",
        "ic_56.",
        "ic1.",
        "ic2.",
        "fc8.",
    )
    missing_backbone = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(rectifier_prefixes)
    ]
    result = {
        "weights_path": str(weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256(weights_path),
        "rectifier": args.rectifier,
        "hst_variant": args.hst_variant if args.rectifier == "hst" else None,
        "converted_key_count": len(converted),
        "missing_keys": incompatible.missing_keys,
        "missing_backbone_keys": missing_backbone,
        "unexpected_keys": incompatible.unexpected_keys,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
