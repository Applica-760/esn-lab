# scripts/test.zsh

echo "[INFO] exec test"

conda activate research

python cli.py --debug train single

python cli.py --debug train batch

python cli.py --debug predict single

python cli.py --debug predict batch

python cli.py --debug evaluate run