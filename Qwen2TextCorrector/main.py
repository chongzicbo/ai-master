import argparse
import sys

from loguru import logger

from model import Qwen2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        default="./grammar/train_sharegpt.jsonl",
        type=str,
        help="Train file",
    )
    parser.add_argument(
        "--test_file",
        default="./grammar/test_sharegpt.jsonl",
        type=str,
        help="Test file",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        type=str,
        help="Transformers model or path",
    )
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run predict."
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 mixed precision training.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs-qwen-1.5b-demo/",
        type=str,
        help="Model output directory",
    )
    parser.add_argument(
        "--prompt_template_name", default="qwen", type=str, help="Prompt template name"
    )
    parser.add_argument(
        "--max_seq_length", default=512, type=int, help="Input max sequence length"
    )
    parser.add_argument(
        "--max_length", default=512, type=int, help="Output max sequence length"
    )
    parser.add_argument(
        "--num_epochs", default=1, type=float, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--eval_steps", default=50, type=int, help="Eval every X steps")
    parser.add_argument(
        "--save_steps", default=50, type=int, help="Save checkpoint every X steps"
    )
    parser.add_argument("--local_rank", type=int, help="Used by dist launchers")
    args = parser.parse_args()
    logger.info(args)
    # fine-tune Llama model
    if args.do_train:
        logger.info("Loading data...")
        model_args = {
            "use_peft": True,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "bf16": args.bf16,
            "prompt_template_name": args.prompt_template_name,
        }
        model = Qwen2Model(args.model_name, args=model_args)
        model.train_model(args.train_file, eval_data=args.test_file)


if __name__ == "__main__":
    main()
