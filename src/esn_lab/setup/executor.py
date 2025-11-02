# esn_lab/setup/executor.py
from omegaconf import OmegaConf
from .registry import REGISTRY
from .config import Config


def _canonical_mode(mode: str) -> str:
    """Normalize CLI mode to schema/config branch name.
    pred -> predict, eval -> evaluate
    """
    return {"pred": "predict", "eval": "evaluate"}.get(mode, mode)

def execute_runner(args, merged_cfg, run_dir):
    regis = REGISTRY.get(args.mode, {}).get("variants", {}).get(args.variant)
    if not regis or "runner" not in regis:
        print(f"[ERROR] Runner for {args.mode}/{args.variant} is not registered.")
        return

    if run_dir:
        merged_cfg["run_dir"] = str(run_dir)
        
    schema = OmegaConf.structured(Config)
    SchemaCls = regis.get("schema")
    if SchemaCls:
        conf_branch = schema.setdefault(_canonical_mode(args.mode), {})
        conf_branch[args.variant] = OmegaConf.structured(SchemaCls)
        
    cfg = OmegaConf.to_object(OmegaConf.merge(schema, merged_cfg))

    runner = regis["runner"]
    print(f"[INFO] mode: {args.mode} is selected")
    print(f"[INFO] run {args.variant} {args.mode}")
    if run_dir:
        print(f"[ARTIFACT] run_dir={str(run_dir)}")
    print("=====================================")
    runner(cfg)