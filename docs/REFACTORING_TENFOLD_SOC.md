# Tenfold訓練の関心の分離リファクタリング

## 概要

tenfold訓練のrunnerとpipelineにおいて、関心の分離を適切に行うためのリファクタリングを実施しました。

## 設計方針

```
┌─────────────────────────────────────────────────────────────┐
│ Runner層: プロセスのオーケストレーション・並列処理の管理      │
│  - タスクのスケジューリング                                  │
│  - 並列実行の制御                                           │
│  - 乱数シード設定                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Pipeline層: データ処理と学習の実体                           │
│  - データ読み込み                                           │
│  - モデル訓練                                               │
│  - 結果の永続化（重み保存・ログ記録）                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Util層: 汎用機能の外部化                                    │
│  - 重みファイル管理（命名・存在確認・スキップ判定）           │
│  - 実行ログ記録                                             │
│  - データファイルマッピング                                  │
└─────────────────────────────────────────────────────────────┘
```

## 主な変更点

### Phase 1: Util層の整理

#### 1.1 新規作成: `pipeline/tenfold_util/execution_log.py`
- `ExecutionLogger`クラスを追加
- 実行時間とメタデータのCSV記録を担当
- スレッドセーフな書き込みを保証

**責務:**
- CSV追記処理
- タイムスタンプ管理
- ログファイルパス管理

#### 1.2 新規作成: `pipeline/tenfold_util/weight_manager.py`
- `WeightManager`クラスを追加
- 重みファイルの管理を一元化

**責務:**
- 重みファイルパスの決定
- 既存ファイルの存在確認
- スキップ判定ロジック（`never`, `if_exists`, `force_retrain`モード）
- 実行すべきタスクリストの生成

#### 1.3 整理: `pipeline/tenfold_util/data.py`
- `read_data_from_csvs`を削除（CSVDataLoaderに統合済み）
- `load_10fold_csv_mapping`のみ残す（後方互換性のため）

### Phase 2: Pipeline層の強化

#### 2.1 修正: `pipeline/train/tenfold_trainer.py`
- コンストラクタに`WeightManager`と`ExecutionLogger`を追加
- `run_one_fold_search`の責務を拡大
  - 重みファイルの保存
  - 実行ログの記録
  - 戻り値を`None`に変更（結果はloggerが自動記録）

**変更前:**
```python
def __init__(self, run_dir: str):
    self._trainer = Trainer(run_dir)

def run_one_fold_search(..., weight_dir: Path) -> tuple[float, str]:
    # 学習処理
    # 重みを保存
    return execution_time, timestamp
```

**変更後:**
```python
def __init__(
    self,
    run_dir: str,
    weight_manager: WeightManager,
    execution_logger: ExecutionLogger,
):
    self._trainer = Trainer(run_dir)
    self._weight_manager = weight_manager
    self._execution_logger = execution_logger

def run_one_fold_search(...) -> None:
    # 学習処理
    # 重みを保存（weight_manager経由）
    # ログを記録（execution_logger経由）
```

### Phase 3: Runner層の簡素化

#### 3.1 修正: `runner/train/tenfold/execution.py`
- `_append_result_to_csv`関数を削除（ExecutionLoggerに統合）
- `execute_tasks`のシグネチャを変更
  - `env`辞書を個別の引数に分解
  - `data_loader`, `weight_manager`, `execution_logger`を明示的に渡す
- `_run_one_fold_search`を簡素化
  - データローダー生成を削除
  - 戻り値を削除（pipeline側で完結）

**変更前:**
```python
def execute_tasks(cfg, env, hp_overrides, hp_tag, tasks_to_run, ...):
    # タスク実行
    execution_time, timestamp = _run_one_fold_search(...)
    result = {"timestamp": timestamp, "hp_tag": hp_tag, ...}
    _append_result_to_csv(result, env["weight_dir"])

def _run_one_fold_search(...) -> tuple[float, str]:
    setup_worker_seed(seed)
    data_loader, _ = create_data_loader_from_config(cfg)  # 毎回生成
    trainer = TenfoldTrainer(cfg.run_dir)
    return trainer.run_one_fold_search(...)
```

**変更後:**
```python
def execute_tasks(
    cfg, data_loader, weight_manager, execution_logger,
    all_letters, hp_overrides, tasks_to_run, ...
):
    # タスク実行（結果記録はpipeline側で完結）
    _run_one_fold_search(...)

def _run_one_fold_search(...) -> None:
    setup_worker_seed(seed)  # Runner層の責務
    trainer = TenfoldTrainer(
        cfg.run_dir, weight_manager, execution_logger
    )
    trainer.run_one_fold_search(...)  # Pipeline層に委譲
```

#### 3.2 修正: `runner/train/tenfold/main.py`
- `_prepare_run_environment`の戻り値を変更
  - `data_loader`, `weight_manager`, `execution_logger`を返す
  - データローダーの生成をrunner層で1回のみ実行
- `_determine_tasks_to_run`を削除
  - `WeightManager.determine_tasks_to_run`に統合
- `_determine_skip_mode`を追加
  - 設定からスキップモードを決定する補助関数
- `run_tenfold`を整理
  - タスク決定ロジックを`weight_manager`に委譲
  - `execute_tasks`呼び出しを明示的な引数で整理

**変更前:**
```python
def _prepare_run_environment(cfg, tenfold_cfg=None):
    data_loader, data_dir = create_data_loader_from_config(...)
    weight_dir = Path(...) / "weights"
    return {
        "weight_dir": weight_dir,
        "data_dir": data_dir,
        "letters": data_loader.get_available_folds(),
    }

def _determine_tasks_to_run(cfg, hp_overrides, letters, weight_dir, ...):
    # 複雑なスキップ判定ロジック
    # 重みファイル名生成、存在確認など
    ...

def run_tenfold(cfg, *, overrides=None, ...):
    env = _prepare_run_environment(cfg, tenfold_cfg)
    tasks_to_run = _determine_tasks_to_run(
        cfg, hp_overrides, env["letters"], env["weight_dir"], ...
    )
    execution.execute_tasks(cfg, env, hp_overrides, hp_tag, tasks_to_run, ...)
```

**変更後:**
```python
def _prepare_run_environment(cfg, tenfold_cfg=None):
    experiment_dir = Path(...) / experiment_name
    weight_dir = experiment_dir / "weights"
    data_loader, _ = create_data_loader_from_config(...)
    weight_manager = WeightManager(weight_dir)
    execution_logger = ExecutionLogger(experiment_dir)
    return {
        "data_loader": data_loader,
        "weight_manager": weight_manager,
        "execution_logger": execution_logger,
        "letters": data_loader.get_available_folds(),
    }

def _determine_skip_mode(tenfold_cfg):
    # シンプルな設定変換のみ
    ...

def run_tenfold(cfg, *, overrides=None, ...):
    env = _prepare_run_environment(cfg, tenfold_cfg)
    skip_mode = _determine_skip_mode(ten_cfg_effective)
    tasks_to_run = env["weight_manager"].determine_tasks_to_run(
        cfg, hp_overrides, env["letters"], skip_mode
    )
    execution.execute_tasks(
        cfg=cfg,
        data_loader=env["data_loader"],
        weight_manager=env["weight_manager"],
        execution_logger=env["execution_logger"],
        all_letters=env["letters"],
        hp_overrides=hp_overrides,
        tasks_to_run=tasks_to_run,
        ...
    )
```

## 改善点

### 1. 責務の明確化
- **Runner**: オーケストレーションと並列制御のみ
- **Pipeline**: データ処理と結果の永続化
- **Util**: 汎用的な機能の提供

### 2. データフローの単純化
```
変更前:
Runner → データローダー生成 → Pipeline → 結果を返す → Runner → CSV書き込み

変更後:
Runner → データローダー生成（1回のみ）
      → Pipeline → 学習・保存・ログ記録（完結）
```

### 3. 再利用性の向上
- `WeightManager`は他のタスク（評価・予測）でも使用可能
- `ExecutionLogger`は汎用的なログ記録に使用可能
- データローダーの重複生成を排除

### 4. テスタビリティの向上
- 各クラスが独立しており、単体テストが容易
- 依存性注入により、モックの利用が容易

### 5. 保守性の向上
- 関心が分離されており、変更の影響範囲が限定的
- 各モジュールの役割が明確で理解しやすい

## 後方互換性

- 既存の設定ファイルはそのまま使用可能
- 既存の重みファイルもそのまま利用可能
- API変更なし（内部実装のみ変更）

## 今後の改善案

1. `ExecutionLogger`のプロセス間共有
   - 現状は各プロセスでLoggerインスタンスを保持
   - 共有メモリやキューを使った集約も検討可能

2. `WeightManager`のキャッシング
   - 存在確認結果のキャッシュで高速化

3. データローダーのマルチプロセス対応
   - 現状はプロセス間でデータローダーを共有（pickle化）
   - より効率的な共有方法の検討

## 確認事項

このリファクタリング後、以下を確認してください:

1. 既存の設定で正常に動作するか
2. 並列実行・逐次実行の両方で動作するか
3. スキップモード（`skip_existing`, `force_retrain`）が正常に動作するか
4. 実行ログが正しく記録されるか
5. 重みファイルが正しく保存されるか
