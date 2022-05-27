def depth_sampling(self):
    from scipy.special import erf
    from scipy.stats import norm
    P_total = erf(self.sampling_range / np.sqrt(2))
    idx_list = np.arange(0, self.n_samples + 1)
    p_list = (1 - P_total)/2 + ((idx_list/self.n_samples) * P_total)
    k_list = norm.ppf(p_list)
    k_list = (k_list[1:] + k_list[:-1])/2
    return list(k_list)