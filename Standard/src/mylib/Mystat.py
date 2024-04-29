import scipy.stats as stats

def calculate_significance(dof, logllh1, logllh2):
    # Calculate the difference in -log likelihoods
    delta_logllh = logllh1 - logllh2
    
    # Calculate the test statistic
    test_statistic = 2 * delta_logllh
    
    # Calculate the p-value using the chi-square distribution
    p_value = 1 - stats.chi2.cdf(test_statistic, dof)
    
    # Calculate the significance level
    significance = stats.norm.ppf(1 - p_value)
    
    return significance