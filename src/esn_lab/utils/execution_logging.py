"""
実行ログの記録ユーティリティ

機械学習実験の実行時間とメタデータをCSVに記録する。
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
from threading import Lock


class ExecutionLogger:
    """実験実行の記録を管理するクラス
    
    実行時間とメタデータをCSVファイルに追記保存する。
    スレッドセーフな書き込みを保証する。
    """

    def __init__(self, log_dir: Path, log_filename: str = "execution_times.csv"):
        """
        Args:
            log_dir: ログファイルを保存するディレクトリ
            log_filename: ログファイル名（デフォルト: execution_times.csv）
        """
        self.log_path = Path(log_dir) / log_filename
        self._lock = Lock()
        
        # ディレクトリが存在しない場合は作成
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_execution(
        self,
        hp_tag: str,
        fold: str,
        execution_time_sec: float,
        timestamp: str | None = None,
    ) -> None:
        """実行結果をCSVに追記する
        
        Args:
            hp_tag: ハイパーパラメータの識別タグ
            fold: fold ID（例: 'a', 'b', ...）
            execution_time_sec: 実行時間（秒）
            timestamp: タイムスタンプ（Noneの場合は現在時刻を使用）
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        record = {
            "timestamp": timestamp,
            "hp_tag": hp_tag,
            "fold": fold,
            "execution_time_sec": execution_time_sec,
        }

        # スレッドセーフな書き込み
        with self._lock:
            self._append_to_csv(record)

    def _append_to_csv(self, record: dict) -> None:
        """CSVファイルに1行追記する（内部メソッド）
        
        Args:
            record: 記録する辞書
        """
        df = pd.DataFrame([record])
        df = df[["timestamp", "hp_tag", "fold", "execution_time_sec"]]

        file_exists = self.log_path.exists()
        df.to_csv(
            self.log_path,
            mode='a',
            header=not file_exists,
            index=False,
            float_format='%.4f'
        )

    def get_log_path(self) -> Path:
        """ログファイルのパスを取得
        
        Returns:
            Path: ログファイルの絶対パス
        """
        return self.log_path.resolve()
