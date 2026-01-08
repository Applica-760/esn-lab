from esn_lab.model.esn import ESN
from esn_lab.pipeline.eval.predictor import pred
from esn_lab.utils.fold_splitter import get_test_fold, get_train_folds
from concurrent.futures import ProcessPoolExecutor


def eval_tenfold(model: ESN, weights_list, tenfold_U, tenfold_D, tenfold_id, mode="test"):
    """
    10fold評価を実行
    """
    if mode == "test":
        data_getter = get_test_fold
    elif mode == "train":
        data_getter = get_train_folds
    else:
        raise ValueError(f"mode must be 'test' or 'train', got '{mode}'")
    
    all_results = []
    for i in range(10):
        model.Output.setweight(weights_list[i])
        U_list, D_list, id_list = data_getter(tenfold_U, tenfold_D, tenfold_id, i)
        
        fold_results = []
        for U, D, sample_id in zip(U_list, D_list, id_list):
            result = pred(model, sample_id, U, D)
            fold_results.append(result)
        
        all_results.append({"fold_index": i, "results": fold_results})
        print(f"fold {i} evaluation finished")
    
    return all_results


def eval_tenfold_parallel(model: ESN, weights_list, tenfold_U, tenfold_D, tenfold_id, mode="test", n_jobs=1):
    """
    10fold評価を並列実行
    """
    if mode == "test":
        data_getter = get_test_fold
    elif mode == "train":
        data_getter = get_train_folds
    else:
        raise ValueError(f"mode must be 'test' or 'train', got '{mode}'")
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for i in range(10):
            model.Output.setweight(weights_list[i])
            U_list, D_list, id_list = data_getter(tenfold_U, tenfold_D, tenfold_id, i)
            
            # 各サンプルの評価を並列実行するためにsubmit
            fold_futures = []
            for U, D, sample_id in zip(U_list, D_list, id_list):
                future = executor.submit(pred, model, sample_id, U, D)
                fold_futures.append(future)
            
            futures.append((i, fold_futures))
        
        # 結果を取得
        for fold_index, fold_futures in futures:
            fold_results = []
            for future in fold_futures:
                result = future.result()
                fold_results.append(result)
            
            all_results.append({"fold_index": fold_index, "results": fold_results})
            print(f"fold {fold_index} evaluation finished")
    
    return all_results