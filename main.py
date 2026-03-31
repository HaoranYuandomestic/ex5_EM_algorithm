import numpy as np
import matplotlib.pyplot as plt


def generate_gmm_data(n_samples=300, random_state=7):
	rng = np.random.default_rng(random_state)
	k = 3
	n_each = n_samples // k

	true_pi = np.array([0.34, 0.33, 0.33])
	true_means = np.array([
		[-3.2, -1.8],
		[2.8, -0.5],
		[0.2, 3.0],
	])
	true_covs = np.array([
		[[0.9, 0.25], [0.25, 0.8]],
		[[0.8, -0.2], [-0.2, 1.0]],
		[[1.0, 0.3], [0.3, 0.9]],
	])

	x_parts = []
	y_parts = []
	for cls in range(k):
		x_cls = rng.multivariate_normal(true_means[cls], true_covs[cls], size=n_each)
		y_cls = np.full(n_each, cls)
		x_parts.append(x_cls)
		y_parts.append(y_cls)

	x = np.vstack(x_parts)
	y = np.concatenate(y_parts)

	remain = n_samples - x.shape[0]
	if remain > 0:
		extra_labels = rng.choice(k, size=remain, p=true_pi)
		extra_data = []
		for lab in extra_labels:
			extra_data.append(rng.multivariate_normal(true_means[lab], true_covs[lab], size=1)[0])
		x = np.vstack([x, np.array(extra_data)])
		y = np.concatenate([y, extra_labels])

	perm = rng.permutation(x.shape[0])
	return x[perm], y[perm]


def multivariate_gaussian_pdf(x, mean, cov):
	n_features = x.shape[1]
	reg = 1e-6 * np.eye(n_features)
	cov_reg = cov + reg

	det_cov = np.linalg.det(cov_reg)
	inv_cov = np.linalg.inv(cov_reg)

	coef = 1.0 / np.sqrt(((2 * np.pi) ** n_features) * det_cov)
	diff = x - mean
	exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
	return coef * np.exp(exponent)


def clustering_accuracy(y_true, y_pred, n_clusters):
	# For K=3, test all permutations to align cluster ids and true labels.
	import itertools

	best = 0.0
	best_map = None
	for perm in itertools.permutations(range(n_clusters)):
		mapping = {pred: true for pred, true in enumerate(perm)}
		mapped = np.array([mapping[p] for p in y_pred])
		acc = np.mean(mapped == y_true)
		if acc > best:
			best = acc
			best_map = mapping
	return float(best), best_map


class GMMEM:
	def __init__(self, n_components=3, max_iter=1000, tol=1e-6, random_state=7):
		self.n_components = n_components
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self.pi_ = None
		self.means_ = None
		self.covs_ = None
		self.log_likelihood_history_ = []

	def _initialize(self, x):
		rng = np.random.default_rng(self.random_state)
		n_samples, n_features = x.shape

		self.pi_ = np.full(self.n_components, 1.0 / self.n_components)
		indices = rng.choice(n_samples, size=self.n_components, replace=False)
		self.means_ = x[indices].copy()

		base_cov = np.cov(x.T) + 1e-3 * np.eye(n_features)
		self.covs_ = np.array([base_cov.copy() for _ in range(self.n_components)])

	def _e_step(self, x):
		n_samples = x.shape[0]
		resp = np.zeros((n_samples, self.n_components))

		for k in range(self.n_components):
			resp[:, k] = self.pi_[k] * multivariate_gaussian_pdf(x, self.means_[k], self.covs_[k])

		row_sum = np.sum(resp, axis=1, keepdims=True)
		row_sum[row_sum == 0] = 1e-12
		resp = resp / row_sum
		return resp

	def _m_step(self, x, resp):
		n_samples, n_features = x.shape
		nk = np.sum(resp, axis=0)
		nk[nk == 0] = 1e-12

		self.pi_ = nk / n_samples
		self.means_ = (resp.T @ x) / nk[:, np.newaxis]

		new_covs = np.zeros((self.n_components, n_features, n_features))
		for k in range(self.n_components):
			diff = x - self.means_[k]
			weighted_diff = resp[:, k][:, np.newaxis] * diff
			cov_k = (weighted_diff.T @ diff) / nk[k]
			cov_k += 1e-6 * np.eye(n_features)
			new_covs[k] = cov_k
		self.covs_ = new_covs

	def _log_likelihood(self, x):
		total_pdf = np.zeros(x.shape[0])
		for k in range(self.n_components):
			total_pdf += self.pi_[k] * multivariate_gaussian_pdf(x, self.means_[k], self.covs_[k])
		total_pdf = np.clip(total_pdf, 1e-300, None)
		return float(np.sum(np.log(total_pdf)))

	def fit(self, x):
		self._initialize(x)
		prev_ll = None

		for iteration in range(1, self.max_iter + 1):
			resp = self._e_step(x)
			self._m_step(x, resp)

			ll = self._log_likelihood(x)
			self.log_likelihood_history_.append(ll)

			if prev_ll is not None and abs(ll - prev_ll) < self.tol:
				print(f"Converged at iteration {iteration}, delta log-likelihood: {abs(ll - prev_ll):.8f}")
				break
			prev_ll = ll

		return self

	def predict_proba(self, x):
		return self._e_step(x)

	def predict(self, x):
		resp = self.predict_proba(x)
		return np.argmax(resp, axis=1)


def plot_raw_data(x, y_true):
	plt.figure(figsize=(7.6, 6.1))
	colors = ["tab:blue", "tab:orange", "tab:green"]
	for cls in range(3):
		mask = y_true == cls
		plt.scatter(x[mask, 0], x[mask, 1], s=24, alpha=0.8, c=colors[cls], label=f"True class {cls}")
	plt.title("Synthetic GMM Data (True Labels)")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.legend()
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig("origin_data.png", dpi=160)
	plt.show()


def plot_cluster_result(x, y_pred, means):
	plt.figure(figsize=(7.6, 6.1))
	colors = ["tab:blue", "tab:orange", "tab:green"]
	for cls in range(3):
		mask = y_pred == cls
		plt.scatter(x[mask, 0], x[mask, 1], s=24, alpha=0.8, c=colors[cls], label=f"Cluster {cls}")

	plt.scatter(
		means[:, 0],
		means[:, 1],
		s=280,
		marker="X",
		c="black",
		linewidths=1.1,
		label="Estimated Means",
	)

	plt.title("EM-GMM Clustering Result")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.legend()
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig("cluster_output.png", dpi=160)
	plt.show()


def plot_log_likelihood_curve(ll_history):
	plt.figure(figsize=(7.3, 4.7))
	iterations = np.arange(1, len(ll_history) + 1)
	plt.plot(iterations, ll_history, color="tab:red", marker="o", markersize=3.8)
	plt.title("Log-Likelihood Curve of EM")
	plt.xlabel("Iteration")
	plt.ylabel("Log-Likelihood")
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig("log_likelihood_curve.png", dpi=160)
	plt.show()


def main():
	# 1) Generate synthetic GMM data.
	x, y_true = generate_gmm_data(n_samples=300, random_state=7)

	# 2) Train GMM by from-scratch EM.
	model = GMMEM(n_components=3, max_iter=1000, tol=1e-6, random_state=7)
	model.fit(x)

	# 3) Predict cluster labels and compute clustering accuracy.
	y_pred = model.predict(x)
	acc, label_mapping = clustering_accuracy(y_true, y_pred, n_clusters=3)

	print("==== From-scratch EM for GMM Clustering ====")
	print(f"Samples: {x.shape[0]}, Features: {x.shape[1]}, Components: {model.n_components}")
	print(f"Iterations used: {len(model.log_likelihood_history_)}")
	print(f"Final log-likelihood: {model.log_likelihood_history_[-1]:.6f}")
	print(f"Clustering accuracy (best permutation): {acc:.4f}")
	print("Label mapping (pred cluster -> true label):")
	print(label_mapping)
	print("Estimated mixture weights:")
	print(model.pi_)
	print("Estimated means:")
	print(model.means_)

	# 4) Visualization.
	plot_raw_data(x, y_true)
	plot_cluster_result(x, y_pred, model.means_)
	plot_log_likelihood_curve(model.log_likelihood_history_)


if __name__ == "__main__":
	main()