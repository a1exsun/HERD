import argparse
from core.OPRTrainer import OnPolicyReplayTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--output-dir", default="output/qwen_opr_ru", type=str)
    parser.add_argument("--reward-type", default="ru", type=str, choices=["ru", "sc"])
    parser.add_argument("--rho", default=0.01, type=float, help='replay ratio')
    args = parser.parse_args()

    trainer = OnPolicyReplayTrainer(args.model_path, args.output_dir, args.reward_type, args.rho)
    trainer.run()


if __name__ == "__main__":
    main()