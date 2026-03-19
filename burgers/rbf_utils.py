import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
import time

class RBFUtils:

    # Define RBF Kernels
    @staticmethod
    def gaussian_rbf(r, epsilon):
        """Gaussian RBF kernel function."""
        return np.exp(-(epsilon * r) ** 2)

    @staticmethod
    def inverse_multiquadric_rbf(r, epsilon):
        """Inverse Multiquadric RBF kernel function."""
        return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

    @staticmethod
    def linear_rbf(r, epsilon):
        """Linear RBF kernel function."""
        return r
    
    @staticmethod
    def multiquadric_rbf(r, epsilon):
        """Multiquadric RBF kernel function."""
        return np.sqrt(1 + (epsilon * r) ** 2)
    
    @staticmethod
    def matern_kernel(r, epsilon):
        """Matérn kernel function with nu=3/2."""
        sqrt3 = np.sqrt(3)
        return (1 + sqrt3 * epsilon * r) * np.exp(-sqrt3 * epsilon * r)
    
    @staticmethod
    def compute_rbf_jacobian_global_gaussian(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        """
        Compute the Jacobian for the Gaussian RBF kernel using a vectorized global approach.

        Parameters:
        - x_normalized: np.ndarray, normalized input sample (1 x dim).
        - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # Shape: (num_train,)
        rbf_values = RBFUtils.gaussian_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute the Jacobian in a vectorized manner
        t0 = time.time()
        # Shape notes:
        # (q_p_train_norm - x_normalized): (num_train, dim)
        # rbf_values[:, np.newaxis]: (num_train, 1)
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p_norm = -2 * (epsilon**2) * rbf_values[:, np.newaxis] * (x_normalized - q_p_train_norm)

        # Now compute jacobian_norm = W_global.T @ Dphi_Dq_p_norm
        # W_global: (num_train, output_dim)
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # Result: (output_dim, dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm

        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # Shape: (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]

        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    
    
    @staticmethod
    def compute_rbf_jacobian_global_multiquadric(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # (num_train,)
        rbf_values = RBFUtils.multiquadric_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute derivatives in a vectorized manner
        t0 = time.time()
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p_norm = epsilon**2 * (x_normalized - q_p_train_norm) / rbf_values[:, np.newaxis]

        # jacobian_norm: (output_dim, dim) from W_global.T (output_dim x num_train) @ Dphi_Dq_p_norm (num_train x dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm
        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust the Jacobian for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]
        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    
    @staticmethod
    def compute_rbf_jacobian_global_imq(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        """
        Compute the Jacobian for the Inverse Multiquadric (IMQ) RBF kernel using a vectorized global approach.

        Parameters:
        - x_normalized: np.ndarray, normalized input sample (1 x dim).
        - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute pairwise distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # (num_train,)
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute the Jacobian in a vectorized manner
        t0 = time.time()
        # (x_normalized - q_p_train_norm): (num_train, dim)
        # (rbf_values**3): (num_train,)
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p_norm = - (epsilon**2) * (rbf_values**3)[:, np.newaxis] * (x_normalized - q_p_train_norm)

        # Combine with W_global
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # jacobian_norm: (output_dim, dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm
        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust the Jacobian for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]
        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    
    @staticmethod
    def compute_rbf_jacobian_global_linear(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        """
        Compute the Jacobian for the Linear RBF kernel using a vectorized global approach.

        Parameters:
        - x_normalized: np.ndarray, normalized input sample (1 x dim).
        - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter (unused for linear kernel, but kept for consistency).
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # (num_train,)
        if echo_level >= 1:
            print(f"Computed distances in {time.time() - t0:.6f} seconds")

        # Step 2: Compute derivatives in a vectorized manner
        t0 = time.time()
        # Initialize Dphi_Dq_p_norm with zeros
        Dphi_Dq_p_norm = np.zeros((num_train, dim))

        # Avoid division by zero: only compute for non-zero distances
        nonzero_mask = dist_to_sample > 1e-12
        Dphi_Dq_p_norm[nonzero_mask] = (x_normalized - q_p_train_norm[nonzero_mask]) / dist_to_sample[nonzero_mask, np.newaxis]

        # Combine with W_global
        # W_global: (num_train, output_dim)
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # Result: (output_dim, dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm

        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]
        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    

    @staticmethod
    def compute_rbf_jacobian_global_matern32(
            x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        """
        Compute the Jacobian for the Matérn kernel with nu=3/2 using a vectorized global approach.

        Matérn 3/2 kernel:
            phi(r) = (1 + sqrt(3)*epsilon*r) * exp(- sqrt(3)*epsilon * r)

        Its derivative wrt r is:
            phi'(r) = - (sqrt(3)*epsilon)^2 * r * exp(- sqrt(3)*epsilon * r).

        After canceling the factor of r with (x - x_i)/r, the gradient wrt x is:
            dphi/dx = - (sqrt(3)*epsilon)^2 * exp(- sqrt(3)*epsilon * r) * (x - x_i),
        which can also be expressed as:
            dphi/dx = - (sqrt(3)*epsilon)^2 * (phi(r)/(1 + sqrt(3)*epsilon*r)) * (x - x_i).

        We then multiply by W_global^T to get the Jacobian in output-space,
        and finally adjust by scaler.scale_ to revert the Min-Max normalization effect.

        Parameters:
        - x_normalized: np.ndarray, shape (1, dim). The normalized query point.
        - q_p_train_norm: np.ndarray, shape (num_train, dim). Normalized training points.
        - W_global: np.ndarray, shape (num_train, output_dim). Precomputed weights.
        - epsilon: float, RBF parameter.
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Verbosity level.

        Returns:
        - jacobian: np.ndarray, shape (output_dim, dim).
        """
        import time
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # (num_train,)
        sqrt3 = np.sqrt(3)
        # phi(r) = (1 + sqrt3 * eps * r) * exp(- sqrt3 * eps * r)
        rbf_values = (1.0 + sqrt3 * epsilon * dist_to_sample) * np.exp(- sqrt3 * epsilon * dist_to_sample)

        if echo_level >= 1:
            print(f"Computed distances and Matern 3/2 RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute the gradient wrt x in a vectorized manner
        t0 = time.time()
        # We want dphi/dx. One approach is using the simplified expression
        # dphi/dr = - (sqrt3*epsilon)^2 * r * exp(- sqrt3*epsilon * r),
        # so dphi/dx = dphi/dr * (x - xi)/r, which reduces to
        # - (sqrt3*epsilon)^2 * exp(- sqrt3*epsilon * r) * (x - xi).
        # Alternatively, using a ratio with rbf_values:
        # dphi/dx = - (sqrt3*epsilon)^2 * [ rbf_values / (1 + sqrt3*epsilon*r) ] * (x - xi).
        # We'll do the latter for clarity:

        A2 = 3.0 * (epsilon**2)  # (sqrt3 * epsilon)^2
        denom = 1.0 + sqrt3 * epsilon * dist_to_sample  # shape (num_train,)
        # shape: (num_train, dim)
        Dphi_Dq_p_norm = - A2 * (rbf_values / denom)[:, np.newaxis] * (x_normalized - q_p_train_norm)

        # Now multiply by W_global^T to get the Jacobian in output space
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm  # shape: (output_dim, dim)

        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # shape: (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]
        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def interpolate_with_rbf_global_gaussian(q_p_sample, q_p_train, W_global, epsilon, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Gaussian RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Gaussian RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.gaussian_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    

    @staticmethod
    def interpolate_with_rbf_global_multiquadric(q_p_sample, q_p_train, W_global, epsilon, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Multiquadric RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Multiquadric RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.multiquadric_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    
    @staticmethod
    def interpolate_with_rbf_global_imq(q_p_sample, q_p_train, W_global, epsilon, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Inverse Multiquadric RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Inverse Multiquadric RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    
    @staticmethod
    def interpolate_with_rbf_global_linear(q_p_sample, q_p_train, W_global, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Linear RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Linear RBF kernel values for the distances
        t0 = time.time()
        rbf_values = dist_to_sample  # Linear kernel: φ(r) = r
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    @staticmethod
    def interpolate_with_rbf_global_matern32(q_p_sample, q_p_train, W_global, epsilon, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Matérn (nu=3/2) RBF interpolation.

        Matérn 3/2 kernel:
            k(r) = (1 + sqrt(3)*epsilon*r) * exp(- sqrt(3)*epsilon * r)

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p) as a 1D array of shape (dim,).
        - q_p_train: Training data for principal modes, shape (num_train, dim).
        - W_global: Precomputed global RBF weights matrix, shape (num_train, num_secondary_modes).
        - epsilon: The width (shape) parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Verbosity level for timing output (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample, shape (num_secondary_modes,).
        """
        import time
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        # Reshape q_p_sample to (1, dim) for the scaler
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Matérn (nu=3/2) RBF kernel values for the distances
        # k(r) = (1 + sqrt(3)*epsilon*r) * exp(- sqrt(3)*epsilon*r)
        t0 = time.time()
        sqrt3 = np.sqrt(3)
        rbf_values = (1.0 + sqrt3 * epsilon * dist_to_sample) * np.exp(- sqrt3 * epsilon * dist_to_sample)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        # W_global has shape (num_train, num_secondary_modes)
        # rbf_values has shape (num_train,)
        # => q_s_pred: (num_secondary_modes,)
        q_s_pred = rbf_values @ W_global
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
