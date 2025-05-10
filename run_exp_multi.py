import os
import sys
from pathlib import Path
from training_IPPO_multi import training, create_environment, CustomWrapper
# from training_SharedPolicy_multi import training, create_environment, CustomWrapper, SharedRewardWrapper

def run_multiple_trainings(num_runs=5, output_dir_str="runs_multi/runs_def", max_iterations=500):
    base_output = Path(output_dir_str).resolve()
    base_output.mkdir(parents=True, exist_ok=True)

    for i in range(num_runs):
        run_dir = base_output / f"run_{i+1}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log_file = run_dir / "log.txt"
        with open(log_file, "w") as f, RedirectStdStreams(stdout=f, stderr=f):

            print(f"=== Run {i+1} ===")

            num_agents = 2
            visual_observation = False
            max_zombies = 4

            env = create_environment(num_agents=num_agents, visual_observation=visual_observation, max_zombies=max_zombies)
            env = CustomWrapper(env)

            # 添加共享奖励包装器
            # env = SharedRewardWrapper(env)

            checkpoint_path = str((run_dir / "checkpoints").resolve())
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

            training(
                env,
                checkpoint_path=checkpoint_path,
                max_iterations=max_iterations,
            )

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
    max_iterations = 750
    num_runs = 5

    output_dir = Path("runs") / "IPPO"
    run_multiple_trainings(
        num_runs=num_runs,
        output_dir_str=str(output_dir),
        max_iterations=max_iterations
    )
    print(f"Learning finished.")