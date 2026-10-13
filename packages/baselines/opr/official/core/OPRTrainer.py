import subprocess, os, shutil, sys


class OnPolicyReplayTrainer:
    def __init__(self, model_path, output_dir, reward_type, rho):
        self.model_path = model_path
        self.adapter_path = None  # [HERO] LoRA: the previous stage's adapter, continued by the next stage
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.reward_type = reward_type
        self.rho = rho
        
        # [HERO] TRACE+17: the eight TRACE tasks in upstream order, then 17 SuperNI tasks
        self.dataset_list = [
            "C-STANCE",
            "FOMC",
            "MeetingBank",
            "Py150",
            "ScienceQA",
            "NumGLUE-cm",
            "NumGLUE-ds",
            "20Minuten",
            "superni-024", "superni-074", "superni-195", "superni-511", "superni-1219", "superni-574",
            "superni-192", "superni-177", "superni-064", "superni-346", "superni-108", "superni-111",
            "superni-1415", "superni-1447", "superni-828", "superni-1728", "superni-217"
        ]
        # [HERO] protocol budget: 32 updates per stage (train.sh passes it as --max_steps), was per-task epochs
        self.epoch_list = ['32'] * 25
        
        self.dataset_size = 500  # [HERO] TRACE+17 training records per task, was 5000
        self.buffer_size = int(self.dataset_size * self.rho)

        self.task_id = 0
        self.task_num = 25  # [HERO] was 8

    def update_model_path(self):
        sub_dir = sorted(os.listdir(self.output_dir))
        latest_dir = os.path.join(self.output_dir, sub_dir[-1])
        result = None
        for name in os.listdir(latest_dir):
            if name.startswith("checkpoint"):
                result = os.path.join(latest_dir, name)
                break
        return result

    def train(self):
        dataset_path = f'data/{self.dataset_list[self.task_id]}/train.jsonl'

        if self.task_id > 0:
            aux_dataset_path = f'data/{self.dataset_list[self.task_id]}/buffer.jsonl'
            # [HERO] a buffer with no records (every per-task quota int(b / (task_id + 1)) is 0) makes ms-swift's
            # dataset loader fail; training on train.jsonl alone is the same as training on it plus an empty buffer
            if os.path.getsize(aux_dataset_path) > 0:
                dataset_path = f'{dataset_path} {aux_dataset_path}'

        command = [
            'bash', 'scripts/train.sh',
            self.model_path, dataset_path, self.epoch_list[self.task_id], self.output_dir,
            self.adapter_path or ''  # [HERO] LoRA continued from the previous stage
        ]

        subprocess.run(command, check=True)

        self.adapter_path = self.update_model_path()  # [HERO] LoRA: was self.model_path (full checkpoint)

    def generate(self):
        command = [
            sys.executable, f"tools/generate_opr_{self.reward_type}.py",
            "--model-path", self.model_path,
            "--task-id", f'{self.task_id}',
            "--buffer-size", f'{self.buffer_size}',
        ]
        subprocess.run(command, check=True)

    def test(self):
        command = [
            sys.executable, "tools/eval.py",
            "--model-path", self.model_path,
            "--task-id", f'{self.task_id}',
        ]
        subprocess.run(command, check=True)

    def run(self):

        while self.task_id < self.task_num:

            if self.task_id == 0:
                if os.path.exists(self.checkpoint_dir):
                    shutil.rmtree(self.checkpoint_dir)
                os.makedirs(self.checkpoint_dir, exist_ok=True)

            self.train()

            self.test()

            if self.task_id < self.task_num - 1:
                self.generate()
            
            self.task_id += 1