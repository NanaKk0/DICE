"""Read one immutable DeepMD PT foundation model's native symbol order."""

from __future__ import annotations

import argparse
import json

from workflow_backends.deepmd_common.type_maps import (
    FOUNDATION_TYPE_MAP_RESULT_PREFIX,
    validated_foundation_model,
    validated_type_map,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    model = validated_foundation_model(args.model)

    from deepmd.infer import DeepPot

    potential = DeepPot(str(model), device="cuda", auto_batch_size=False)
    type_map = validated_type_map(potential.get_type_map(), role="foundation native")
    print(
        FOUNDATION_TYPE_MAP_RESULT_PREFIX
        + json.dumps(type_map, separators=(",", ":"), allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
