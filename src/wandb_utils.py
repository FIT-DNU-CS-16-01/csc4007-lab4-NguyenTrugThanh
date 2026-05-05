from __future__ import annotations


def init_wandb(args, audit: dict, vocab_size: int):
    if not getattr(args, "use_wandb", False) or getattr(args, "wandb_mode", "disabled") == "disabled":
        return None
    try:
        import wandb
        config = vars(args).copy()
        config.update({"vocab_size_actual": vocab_size, "sequence_audit": audit})
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            mode=args.wandb_mode,
            config=config,
        )
    except Exception as exc:
        print(f"[wandb] disabled because initialization failed: {exc}")
        return None


def log_epoch(run, row: dict) -> None:
    if run is not None:
        run.log(row, step=row.get("epoch"))


def safe_finish(run) -> None:
    if run is not None:
        run.finish()
