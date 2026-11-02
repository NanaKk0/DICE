"""Formal-frame evaluation in the selected sampler Python environment."""
import argparse
from pathlib import Path

from workflow_backends.lammps.identity_swap import finalize
from workflow_backends.sample_artifacts import load_task_context


def evaluate_task(root, *, failure=None, provider=None):
    root = Path(root)
    context = load_task_context(root/'task_context.json')
    if provider is None:
        from workflow_backends.widom.evaluation import DeepMDEvaluationProvider
        provider = DeepMDEvaluationProvider()
    try:
        evaluator = provider.open(tuple(root/name for name in context.ordered_models), context.md_mc['type_map'])
    except Exception as exc:
        return finalize(root, failure=f'{failure or ""}; model initialization: {exc}')
    try:
        return finalize(root, failure=failure, evaluator=evaluator)
    finally:
        evaluator.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', required=True)
    parser.add_argument('--failure')
    args = parser.parse_args()
    evaluate_task(args.task_dir, failure=args.failure)


if __name__ == '__main__':
    main()
