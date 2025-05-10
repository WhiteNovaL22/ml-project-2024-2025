import os
import sys
from pathlib import Path
from training_PPO_single import training, create_environment, CustomWrapper

def run_multiple_trainings(num_runs=5, lr=1e-4, clip_param=0.2, output_dir_str="runs/runs_def", max_iterations=500):
    base_output = Path(output_dir_str).resolve()
    base_output.mkdir(parents=True, exist_ok=True)

    for i in range(num_runs):
        run_dir = base_output / f"run_{i+1}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log_file = run_dir / "log.txt"
        with open(log_file, "w") as f, RedirectStdStreams(stdout=f, stderr=f):

            print(f"=== Run {i+1} | lr: {lr} | clip_param: {clip_param} ===")

            num_agents = 1
            visual_observation = False
            max_zombies = 4

            env = create_environment(num_agents=num_agents, visual_observation=visual_observation, max_zombies=max_zombies)
            env = CustomWrapper(env)

            checkpoint_path = str((run_dir / "checkpoints").resolve())
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

            training(
                env,
                checkpoint_path=checkpoint_path,
                max_iterations=max_iterations,
                lr=lr,
                clip_param=clip_param  # 传入 clip 参数
            )

        print(f"run_{i+1} finished.")

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr

    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

if __name__ == "__main__":
    # 参数扫描设置
    ref_lr = 3e-4
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    ref_clip = 0.3
    clip_params = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5]
    max_iterations = 750
    num_runs = 5

    output_dir = Path("runs") / f"lr_{ref_lr:.0e}__clip_{ref_clip:.1f}"
    run_multiple_trainings(
        num_runs=num_runs,
        lr=ref_lr,
        clip_param=ref_clip,
        output_dir_str=str(output_dir),
        max_iterations=max_iterations
    )
    print(f"Learning fin: lr = {ref_lr}, clip = {ref_clip}")

    # # 固定 clip_param，扫描 lr
    # for lr in learning_rates:
    #     output_dir = Path("runs") / f"lr_{lr:.0e}__clip_{ref_clip:.1f}"
    #     run_multiple_trainings(
    #         num_runs=num_runs,
    #         lr=lr,
    #         clip_param=ref_clip,
    #         output_dir_str=str(output_dir),
    #         max_iterations=max_iterations
    #     )
    #     print(f"Learning fin: lr = {lr}")
    #
    # # 固定 lr，扫描 clip_param
    # for clip in clip_params:
    #     output_dir = Path("runs") / f"lr_{ref_lr:.0e}__clip_{clip:.1f}"
    #     run_multiple_trainings(
    #         num_runs=num_runs,
    #         lr=ref_lr,
    #         clip_param=clip,
    #         output_dir_str=str(output_dir),
    #         max_iterations=max_iterations
    #     )
    #     print(f"Learning fin: clip = {clip}")

    print(f"Learning finished.")