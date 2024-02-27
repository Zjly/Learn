import numpy as np
import pandas as pd


def is_pareto_efficient(costs):
    """
    寻找帕累托有效点
    :param costs: 一个 (n_points, n_costs) 的数组
    :return: 一个 (n_points, ) 的布尔数组，指示每个点是否为帕累托有效点
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] <= c, axis=1)) and np.all(np.any(costs[i + 1:] <= c, axis=1))
    return is_efficient


def read(type, top_k, standard):
    df = pd.read_excel(f"{type}.xlsx")

    arrays = [standard]
    for index, row in df.iterrows():
        if int(row["top-k"]) != top_k:
            continue

        bleu = row['BLEU']
        meteor = row['meteor']
        rouge = row['rouge']
        levenshtein_distance = row['levenshtein distance']
        jaro_winkler = row['jaro winkler']
        coverage = row['coverage']
        tail_coverage = row['tail coverage']

        arrays.append([bleu, meteor, rouge, levenshtein_distance, jaro_winkler, coverage, tail_coverage])

    data = np.array(arrays)
    pareto_front = data[is_pareto_efficient(data)]
    print(len(data), len(pareto_front))
    # print(pareto_front)


if __name__ == '__main__':
    read("java", 5, [0.8402, 0.8687, 0.9138, 0.4979, 0.3922, 0.2715, 0.1289])
    read("java", 10, [0.8706, 0.8895, 0.9287, 0.5386, 0.4150, 0.3271, 0.1805])
