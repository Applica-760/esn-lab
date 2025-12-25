# esn_lab/cli.py
from esn_lab.setup.workspace import initialize_configs, setup_rundir
from esn_lab.setup.args import parse_args
from esn_lab.setup.config_loader import load_and_merge_configs
from esn_lab.setup.executor import execute_runner

def main():
    print("=====================================")
    args = parse_args()

    # モードに応じた処理の振り分け
    if args.mode == "init":
        initialize_configs()
        return

    # 1. 設定ファイルの読み込みとマージ
    merged_cfg = load_and_merge_configs(args.mode, args.variant)
    
    # 2. 実行ディレクトリのセットアップ
    run_dir = setup_rundir(args.mode, args.variant, args.debug, merged_cfg)
    print("=====================================")
    
    # 3. ランナーの実行
    execute_runner(args, merged_cfg, run_dir)


if __name__ == "__main__":
    main()