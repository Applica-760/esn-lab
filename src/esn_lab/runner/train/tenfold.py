from esn_lab.model.esn import ESN
from esn_lab.pipeline.train.trainer import train

def run_tenfold(model: ESN, optimizer, tenfold_U_list, tenfold_D_list):
    weights_list = []
    
    for i in range(10):
        # i番目を除外して、残りの9つのデータ群を連結して1つのデータ群に統合
        U_list = []
        D_list = []
        for j in range(10):
            if j != i:
                U_list.extend(tenfold_U_list[j])
                D_list.extend(tenfold_D_list[j])
        
        # trainer pipelineに投入
        output_weight = train(model, optimizer, U_list, D_list)
        weights_list.append(output_weight)

        print(f"fold {i} id finished")
    
    return weights_list