# utils/fold_splitter.py


def get_test_fold(tenfold_U, tenfold_D, tenfold_id, fold_index):
    """
    テスト性能評価用：i番目のデータ群のみを返す
    """
    U_list = tenfold_U[fold_index]
    D_list = tenfold_D[fold_index]
    id_list = tenfold_id[fold_index]
    return U_list, D_list, id_list


def get_train_folds(tenfold_U, tenfold_D, tenfold_id, fold_index):
    """
    訓練性能評価用：i番目以外の9個のデータ群を連結して返す
    """
    U_list = []
    D_list = []
    id_list = []
    for j in range(10):
        if j != fold_index:
            U_list.extend(tenfold_U[j])
            D_list.extend(tenfold_D[j])
            id_list.extend(tenfold_id[j])
    return U_list, D_list, id_list
