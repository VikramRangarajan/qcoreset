import os
import sys
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Callable

import dotenv

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.absolute()
EXPERIMENTS: dict[int | str, Callable] = {}


def exp(exp_num: int):
    def _func(f: Callable[[], None]):
        f_name = f.__name__  # type: ignore
        if exp_num in EXPERIMENTS or f_name in EXPERIMENTS:
            raise ValueError(f"Duplicate experiment: {exp_num} ({f_name})")

        def _print() -> None:
            print(f"Launching Experiment {f_name} ({exp_num})")
            f()

        EXPERIMENTS[exp_num] = _print
        EXPERIMENTS[str(exp_num)] = _print
        EXPERIMENTS[f_name] = _print
        return _print

    return _func


def dict_to_flags(flags: dict):
    flags_list = [f"--{k}={v}" for k, v in flags.items()]
    return " ".join(flags_list)


def submit_job(
    flags: str | dict,
    name: str,
    time: str = "04:00:00",
    partition: str = "a10",
    qos: str = "standby",
    gpus: int = 1,
    nodes: int = 1,
):
    if isinstance(flags, dict):
        flags = dict_to_flags(flags)
    slurm_script = dedent(
        f"""\
        #!/bin/bash

        #SBATCH -n {nodes}
        #SBATCH -c 16
        #SBATCH -t {time}
        #SBATCH -J {name}
        #SBATCH -q {qos}
        #SBATCH -A {os.environ["SLURM_ACCOUNT"]}

        #SBATCH --mem=16G

        #SBATCH -p {partition}
        #SBATCH --gres=gpu:{gpus}

        . ~/.bashrc
        module purge

        cd {PROJECT_ROOT}
        uv run train.py {flags}
        """
    )
    with NamedTemporaryFile(suffix=f"{name}.sh", delete=False) as f:
        pass

    Path(f.name).write_text(slurm_script)
    run(["sbatch", f.name])


@exp(-100)
def example():
    submit_job({"compile": True, "batch_size": 128}, "test")


@exp(1)
def baseline():
    submit_job(
        {
            "selection_method": "random_full",
            "dataset": "cifar10",
            "arch": "resnet20",
            "corrupt_ratio": 0.1,
            "epochs": 10,
        },
        "baseline",
    )


def main():
    experiments = sys.argv[1:]
    for exp in experiments:
        if exp not in EXPERIMENTS:
            raise ValueError(
                f"Invalid experiment: got {exp}, expected one of {list(EXPERIMENTS.keys())}"
            )
        EXPERIMENTS[exp]()


if __name__ == "__main__":
    main()
