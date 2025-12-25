# runner/predict.py
import numpy as np
from pathlib import Path

from esn_lab.setup.config import Config
from esn_lab.pipeline.pred.predictor import Predictor
from esn_lab.pipeline.data import create_data_loader_from_config
from esn_lab.model.model_builder import get_model
from esn_lab.utils.data_processing import make_onehot
from esn_lab.utils.io import save_json, to_keyed_dict
from esn_lab.utils.constants import PREDICT_RECORD_FILE


def _find_sample_in_loader(data_loader, fold_ids: list[str], target_id: str) -> tuple[np.ndarray, int] | None:
    """data loaderから指定されたsample_idのデータを検索する
    
    Args:
        data_loader: データローダーインスタンス
        fold_ids: 検索対象のfold ID一覧
        target_id: 検索するサンプルID
    
    Returns:
        tuple[np.ndarray, int] | None: (sequence, class_id) または None（見つからない場合）
    """
    for sample_id, U, class_id in data_loader.load_fold_data(fold_ids):
        if sample_id == target_id:
            return U, class_id
    return None


def single_predict(cfg: Config):
    """単一サンプルの予測を実行
    
    設定に data_source が指定されていればdata loader経由で読み込み、
    path が指定されていれば直接画像ファイルから読み込む（後方互換）。
    """
    # model定義
    model, optimizer = get_model(cfg)

    # load weight
    weight_dir = Path(cfg.predict.single.weight)
    weight_path = sorted(weight_dir.glob("*.npy"))[0]
    weight = np.load(weight_path, allow_pickle=True)
    model.Output.setweight(weight)
    print(f"[ARTIFACT] {weight_path} is loaded")

    single_cfg = cfg.predict.single
    
    # data_source が指定されている場合は data loader 経由で読み込み
    if hasattr(single_cfg, "data_source") and single_cfg.data_source is not None:
        data_loader, _ = create_data_loader_from_config(cfg, single_cfg)
        all_folds = data_loader.get_available_folds()
        
        target_id = single_cfg.id
        result_data = _find_sample_in_loader(data_loader, all_folds, target_id)
        
        if result_data is None:
            raise ValueError(f"Sample ID '{target_id}' not found in any fold")
        
        U, class_id = result_data
        print(f"[INFO] Loaded sample '{target_id}' from data loader")
    
    # 後方互換: path が指定されている場合は直接画像を読み込み
    elif hasattr(single_cfg, "path") and single_cfg.path:
        import cv2
        U = cv2.imread(single_cfg.path, cv2.IMREAD_UNCHANGED).T
        class_id = single_cfg.class_id
        print(f"[INFO] Loaded sample from image file: {single_cfg.path}")
    
    else:
        raise ValueError(
            "Config 'predict.single' requires either 'data_source' or 'path' to be specified"
        )

    predict_len = len(U)
    D = make_onehot(class_id, predict_len, cfg.model.Ny)

    # set predictor
    predictor = Predictor(cfg.run_dir)
    result = predictor.predict(model, single_cfg.id, U, D)
    print("=====================================")

    # log output layer
    save_json(to_keyed_dict(result), cfg.run_dir, PREDICT_RECORD_FILE)
    print("[INFO] output layer saved to json\n=====================================")

    return




def batch_predict(cfg: Config):
    """複数サンプルのバッチ予測を実行
    
    設定に data_source が指定されていればdata loader経由で読み込み、
    paths が指定されていれば直接画像ファイルから読み込む（後方互換）。
    """
    # model定義
    model, optimizer = get_model(cfg)

    # load weight
    weight_dir = Path(cfg.predict.batch.weight)
    weight_path = sorted(weight_dir.glob("*.npy"))[0]
    weight = np.load(weight_path)
    model.Output.setweight(weight)
    print(f"[ARTIFACT] {weight_path} is loaded")

    batch_cfg = cfg.predict.batch
    
    # set predictor
    predictor = Predictor(cfg.run_dir)
    results = {}
    
    # data_source が指定されている場合は data loader 経由で読み込み
    if hasattr(batch_cfg, "data_source") and batch_cfg.data_source is not None:
        data_loader, _ = create_data_loader_from_config(cfg, batch_cfg)
        all_folds = data_loader.get_available_folds()
        
        for target_id, expected_class_id in zip(batch_cfg.ids, batch_cfg.class_ids):
            result_data = _find_sample_in_loader(data_loader, all_folds, target_id)
            
            if result_data is None:
                print(f"[WARN] Sample ID '{target_id}' not found in any fold. Skipping.")
                continue
            
            U, class_id = result_data
            
            # class_id の検証（設定値とデータの不一致チェック）
            if class_id != expected_class_id:
                print(
                    f"[WARN] Class ID mismatch for sample '{target_id}': "
                    f"config={expected_class_id}, data={class_id}. Using data value."
                )
            
            predict_len = len(U)
            D = make_onehot(class_id, predict_len, cfg.model.Ny)
            result = to_keyed_dict(predictor.predict(model, target_id, U, D))
            results.update(result)
            print(f"[INFO] Predicted sample '{target_id}' from data loader")
    
    # 後方互換: paths が指定されている場合は直接画像を読み込み
    elif hasattr(batch_cfg, "paths") and batch_cfg.paths:
        import cv2
        for i in range(len(batch_cfg.ids)):
            U = cv2.imread(batch_cfg.paths[i], cv2.IMREAD_UNCHANGED).T
            predict_len = len(U)
            D = make_onehot(batch_cfg.class_ids[i], predict_len, cfg.model.Ny)
            result = to_keyed_dict(predictor.predict(model, batch_cfg.ids[i], U, D))
            results.update(result)
            print(f"[INFO] Predicted sample '{batch_cfg.ids[i]}' from image file")
    
    else:
        raise ValueError(
            "Config 'predict.batch' requires either 'data_source' or 'paths' to be specified"
        )

    print("=====================================")

    # log output layer
    save_json(results, cfg.run_dir, PREDICT_RECORD_FILE)
    print("[INFO] output layer saved to json\n=====================================")

    return


# 重みを変えてループ回しながら，batch_predictionを呼び出して
