import scipy.stats as stats
import numpy as np

def calculate_significance(dof, logllh1, logllh2):
    # Calculate the difference in -log likelihoods
    delta_logllh = logllh1 - logllh2
    
    # Calculate the test statistic
    test_statistic = 2 * delta_logllh
    
    # Calculate the p-value using the chi-square distribution
    p_value = 1 - stats.chi2.cdf(test_statistic, dof)
    
    # Calculate the significance level
    significance = stats.norm.ppf(1 - p_value/2)
    
    return significance

def calculate_reduced_chi2(observed, predicted, errors, num_params):
    """
    计算还原卡方（reduced chi-square）值。
    
    参数:
    - observed: 观测数据值的数组
    - predicted: 模型预测值的数组
    - errors: 观测数据的不确定度（标准差）的数组
    - num_params: 模型参数的数量
    
    返回:
    - reduced_chi2: 计算得到的还原卡方值
    """
    # 确保输入是 NumPy 数组
    observed = np.array(observed)
    predicted = np.array(predicted)
    errors = np.array(errors)
    
    # 计算卡方统计量
    chi2 = np.sum(((observed - predicted) / errors) ** 2)
    
    # 自由度 = 数据点数 - 模型参数数
    degrees_of_freedom = len(observed) - num_params
    
    # 计算还原卡方值
    reduced_chi2 = chi2 / degrees_of_freedom

    print(f"Reduced chi-square: {chi2}/{degrees_of_freedom} = {reduced_chi2}")
    
    return reduced_chi2