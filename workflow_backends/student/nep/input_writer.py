from __future__ import annotations

from pathlib import Path


def _zbl_cutoff(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not bool(value.get("enabled", True)):
            return None
        if value.get("outer_cutoff") is not None:
            return float(value["outer_cutoff"])
        if value.get("cutoff") is not None:
            return float(value["cutoff"])
        return None
    return float(value)


def _architecture_lines(nep_cfg: dict) -> list[str]:
    cutoff = dict(nep_cfg.get("cutoff", {}) or {})
    n_max = dict(nep_cfg.get("n_max", {}) or {})
    basis_size = dict(nep_cfg.get("basis_size", {}) or {})
    lines = [
        f"version      {int(nep_cfg.get('version', 4))}",
        f"cutoff       {float(cutoff.get('radial', 7.0))} {float(cutoff.get('angular', 5.0))}",
        f"n_max        {int(n_max.get('radial', 8))} {int(n_max.get('angular', 4))}",
        f"basis_size   {int(basis_size.get('radial', 8))} {int(basis_size.get('angular', 8))}",
        f"l_max        {str(nep_cfg.get('l_max', '4 2 0'))}",
        f"neuron       {int(nep_cfg.get('neuron', 80))}",
    ]
    zbl = _zbl_cutoff(nep_cfg.get("zbl"))
    if zbl is not None:
        lines.insert(1, f"zbl          {zbl}")
    return lines


def write_nep_input(
    output_path: str | Path,
    nep_cfg: dict,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    type_map = [str(item) for item in nep_cfg.get("type_map", [])]
    generation = int(nep_cfg.get("generation", 100000))
    save_potential_every = int(nep_cfg.get("save_potential_every") or max(1, generation))
    save_potential_format = int(nep_cfg.get("save_potential_format", 0))
    save_potential_save_restart = int(nep_cfg.get("save_potential_save_restart", 1))
    fine_tune_model = nep_cfg.get("fine_tune_model_path")
    fine_tune_restart = nep_cfg.get("fine_tune_restart_path")
    fine_tune_lines = []
    if fine_tune_model or fine_tune_restart:
        if not fine_tune_model or not fine_tune_restart:
            raise ValueError("NEP fine_tune requires both fine_tune_model_path and fine_tune_restart_path")
        fine_tune_lines = [f"fine_tune    {fine_tune_model} {fine_tune_restart}"]

    lines = [
        f"type         {len(type_map)} {' '.join(type_map)}",
        *_architecture_lines(nep_cfg),
        *fine_tune_lines,
        *([] if nep_cfg.get("lambda_1") is None else [f"lambda_1     {float(nep_cfg.get('lambda_1'))}"]),
        *([] if nep_cfg.get("lambda_2") is None else [f"lambda_2     {float(nep_cfg.get('lambda_2'))}"]),
        f"lambda_e     {float(nep_cfg.get('lambda_e', 1.0))}",
        f"lambda_f     {float(nep_cfg.get('lambda_f', 1.0))}",
        f"lambda_v     {float(nep_cfg.get('lambda_v', 0.1))}",
        f"batch        {int(nep_cfg.get('batch', 1000))}",
        f"population   {int(nep_cfg.get('population', 50))}",
        f"generation   {generation}",
        f"save_potential {save_potential_every} {save_potential_format} {save_potential_save_restart}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path.resolve()


def write_nep_prediction_input(output_path: str | Path, nep_cfg: dict) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    type_map = [str(item) for item in nep_cfg.get("type_map", [])]
    lines = [
        f"type         {len(type_map)} {' '.join(type_map)}",
        *_architecture_lines(nep_cfg),
        "prediction   1",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path.resolve()
