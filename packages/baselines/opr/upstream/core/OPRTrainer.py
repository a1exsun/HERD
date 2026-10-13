import subprocess, os, shutil, sys


class OnPolicyReplayTrainer:
    def __init__(self, model_path, output_dir, reward_type, rho):
        self.model_path = model_path
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.reward_type = reward_type
        self.rho = rho
        
        self.dataset_list = [
            "C-STANCE",
            "FOMC",
            "MeetingBank",
            "Py150",
            "ScienceQA",
            "NumGLUE-cm",
            "NumGLUE-ds",
            "20Minuten"
        ]
        self.epoch_list = ['5', '3', '7', '5', '3', '5', '5', '7']
        
        self.dataset_size = 5000
        self.buffer_size = int(self.dataset_size * self.rho)

        self.task_id = 0
        self.task_num = 8

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
            dataset_path = f'{dataset_path} {aux_dataset_path}'

        command = [
            'bash', 'scripts/train.sh',
            self.model_path, dataset_path, self.epoch_list[self.task_id], self.output_dir
        ]

        subprocess.run(command, check=True)

        self.model_path = self.update_model_path()

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