import numpy as np


def sample(self, n_samples, burn_in=100, thin=1, warm_start=50,
           verbose=True, store_g=True):
    if self.include_skewness and warm_start > 0:
        if verbose:
            print(f"Warm start (skewness off): {warm_start} iterations")
        self.delta = np.zeros(self.J_minus_1)
        self.g = np.full(self.N, self.mu_G)
        saved_flag = self.include_skewness
        self.include_skewness = False
        for _ in range(warm_start):
            self.gibbs_step()
        self.include_skewness = saved_flag
        if getattr(self, 'g_dist', 'halfnormal') == 'exponential':
            self.g = np.random.exponential(scale=1.0, size=self.N)
        else:
            self.g = np.abs(np.random.randn(self.N))
        self.delta = np.random.normal(0.0, 1.5, size=self.J_minus_1)
        if getattr(self, "constrain_skewness", False):
            self.delta[0] = abs(self.delta[0])

    samples = {'beta': np.zeros((n_samples, self.J_minus_1, self.p)),
               'Psi': np.zeros((n_samples, self.J_minus_1, self.J_minus_1))}
    if self.include_skewness:
        samples['delta'] = np.zeros((n_samples, self.J_minus_1))
        if store_g:
            samples['g'] = np.zeros((n_samples, self.N))
    if self.J_minus_2 > 0:
        samples['gamma'] = np.zeros((n_samples, self.J_minus_2))

    if verbose:
        print(f"Burn-in: {burn_in} iterations")
    for it in range(burn_in):
        self.gibbs_step()
        if verbose and (it + 1) % max(1, burn_in // 5) == 0:
            print(f"  Burn-in: {it + 1}/{burn_in}")

    if verbose:
        print(f"Sampling: {n_samples} draws (thin={thin})")
    collected, it = 0, 0
    max_it = n_samples * thin + 10
    while collected < n_samples and it < max_it:
        self.gibbs_step()
        it += 1
        if it % thin == 0:
            samples['beta'][collected] = self.beta
            samples['Psi'][collected] = self.Psi
            if self.include_skewness:
                samples['delta'][collected] = self.delta
                if store_g:
                    samples['g'][collected] = self.g
            if self.J_minus_2 > 0:
                samples['gamma'][collected] = self.gamma
            collected += 1
            if verbose and collected % max(1, n_samples // 5) == 0:
                print(f"  Samples: {collected}/{n_samples}")

    if collected < n_samples:
        for key in samples:
            samples[key] = samples[key][:collected]

    for key, arr in samples.items():
        a = np.asarray(arr)
        if a.dtype != np.float64:
            samples[key] = np.asarray(a.tolist() if a.dtype == object else a, dtype=np.float64)

    return samples
