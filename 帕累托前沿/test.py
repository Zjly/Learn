import numpy as np

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

# 假设你的数据
data = np.array([[0.3, 0.4, 0.5], [0.2, 0.6, 0.4], [0.5, 0.3, 0.6], [0.1, 0.1, 0.1]])

# 找到帕累托前沿点
pareto_front = data[is_pareto_efficient(data)]

print("帕累托前沿点：", pareto_front)
