
import json
import os

from projects.utils.plot.plot_prediction import plot_prediction


def sort_id():
    """
    引数：
    ・json
    ・条件の情報など
    
    常に判別に失敗している，成功しているID
    特定のラベルのIDのみ抽出するなどのロジック
    将来的に実装

    戻り値：
    ・求めた条件に合致するIDのリスト
    """

    return


def main():
    """
    コマンドライン引数：
    ・jsonのパスかディレクトリなど

    jsonのロード
    上のソート関数か，直接文字列指定などでプロット対象をピックアップ

    JSONの直接のデータ構造
    [
        {
            fold_index: foldのid
            results:[
                {
                "id": データのid,
                "predictions": np.array(),
                "labels": np.array(),
                }, ...（1つのfoldのデータ数分繰り返し）
            ]
        }, ...(foldの数だけ繰り返し)
    ]
    

    ＊plot predictionの呼び出し（渡すのは特定IDのresultsリスト）
    """
    # JSONファイルのパス
    json_path = "outputs/experiments/eval_results/a/Nx7_dens0.5_inscl0.001_rho0.9/test_results.json"
    
    # JSONを読み込み
    with open(json_path, "r") as f:
        all_results = json.load(f)
    
    # プロット対象のID（ベタ書き）
    target_id = "08_013319"
    
    # fold_index=0のresultsを使用
    fold_data = all_results[0]
    results = fold_data["results"]
    
    # 保存先ディレクトリ
    save_dir = "outputs/analysis/prediction_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # プロット実行
    save_path = os.path.join(save_dir, f"prediction_{target_id}")
    plot_prediction(results, target_id, save_path, ext="png")


if __name__ == "__main__":
    main()