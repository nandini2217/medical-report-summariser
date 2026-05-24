"""
scripts/train.py
----------------
Entry point to start fine-tuning the model.

Local CPU (slow, just to verify it works):
    python scripts/train.py --smoke_test

Google Colab GPU (real training):
    python scripts/train.py

"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main(args):
    print("=" * 55)
    print("  Medical Report Summariser — Training")
    print("=" * 55)

    # Smoke test mode: shrink everything so it runs in 60 seconds
    if args.smoke_test:
        print("\n[SMOKE TEST MODE] Shrinking config for quick local run...")
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        cfg["training"]["epochs"]        = 1
        cfg["training"]["batch_size"]    = 1
        cfg["training"]["logging_steps"] = 1
        cfg["training"]["eval_steps"]    = 5
        cfg["training"]["save_steps"]    = 5
        cfg["data"]["max_input_length"]  = 128
        cfg["data"]["max_target_length"] = 64

        # Use tiny model for smoke test
        cfg["model"]["base_model"] = "facebook/bart-base"

        with open("config_smoke.yaml", "w") as f:
            yaml.dump(cfg, f)

        from src.model.trainer import train
        train("config_smoke.yaml")

    else:
        from src.model.trainer import train
        train("config.yaml")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke_test", action="store_true",
                   help="Quick run to verify everything works (1 epoch, tiny model)")
    main(p.parse_args())
