from __future__ import annotations

from pathlib import Path


def build_dp_train_command(
    dp_command: list[str],
    input_json_name: str,
    *,
    init_model: str | Path | None = None,
    finetune_model: str | Path | None = None,
    finetune_use_pretrain_script: bool = False,
    finetune_model_branch: str | None = None,
) -> list[str]:
    if init_model is not None and finetune_model is not None:
        raise ValueError("DeepMD training cannot use --init-model and --finetune in the same run")
    train_cmd = list(dp_command) + [str(input_json_name)]
    if finetune_model is not None:
        train_cmd.extend(["--finetune", str(Path(finetune_model).resolve())])
        if finetune_model_branch:
            train_cmd.extend(["--model-branch", str(finetune_model_branch)])
        if finetune_use_pretrain_script:
            train_cmd.append("--use-pretrain-script")
    elif init_model is not None:
        train_cmd.extend(["--init-model", str(Path(init_model).resolve())])
    return train_cmd
