from esn_lab.model.esn import ESN
from esn_lab.pipeline.train.trainer import train
from esn_lab.utils.fold_splitter import get_train_folds
from concurrent.futures import ProcessPoolExecutor


def train_tenfold(model: ESN, optimizer, tenfold_U, tenfold_D, tenfold_id):
    weights_list = []
    
    for i in range(10):
        U_list, D_list, _ = get_train_folds(tenfold_U, tenfold_D, tenfold_id, i)
        
        # trainer pipelineに投入
        output_weight = train(model, optimizer, U_list, D_list)
        weights_list.append(output_weight)

        print(f"fold {i} finished")
    
    return weights_list


def train_tenfold_parallel(model: ESN, optimizer, tenfold_U, tenfold_D, tenfold_id, n_jobs=1):
    weights_list = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for i in range(10):
            U_list, D_list, _ = get_train_folds(tenfold_U, tenfold_D, tenfold_id, i)
            
            # trainer pipelineに投入
            future = executor.submit(train, model, optimizer, U_list, D_list)
            futures.append(future)
        
        for i, future in enumerate(futures):
            output_weight = future.result()
            weights_list.append(output_weight)
            print(f"fold {i} finished")
    
    return weights_list