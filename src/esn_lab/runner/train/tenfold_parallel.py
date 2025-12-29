from esn_lab.model.esn import ESN
from esn_lab.pipeline.train.trainer import train
from concurrent.futures import ProcessPoolExecutor

def run_tenfold_parallel(model: ESN, optimizer, tenfold_U_list, tenfold_D_list, n_jobs=1):
    weights_list = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for i in range(10):
            # i番目を除外して、残りの9つのデータ群を連結して1つのデータ群に統合
            U_list = []
            D_list = []
            for j in range(10):
                if j != i:
                    U_list.extend(tenfold_U_list[j])
                    D_list.extend(tenfold_D_list[j])
            
            # trainer pipelineに投入
            future = executor.submit(train, model, optimizer, U_list, D_list)
            futures.append(future)
        
        for i, future in enumerate(futures):
            output_weight = future.result()
            weights_list.append(output_weight)
            print(f"fold {i} id finished")
    
    return weights_list