from qerasure_python import NoiseParams as _CPP_NoiseParams
from qerasure_python import build_noise_model as _CPP_build_noise_model

class NoiseParams:
    """
    A Python wrapper around the C++ NoiseModel class.

    This class provides a Python interface to the C++ implementation of noise models for quantum error correction.
    It initializes the noise model with given parameters and exposes properties such as the error probabilities for different types of errors.
    """
    def __init__(self, p_single_qubit_depolarize = 0.0, 
                 p_two_qubit_depolarize = 0.0, 
                 p_measurement_error = 0.0, 
                 p_single_qubit_erasure = 0.0, 
                 p_two_qubit_erasure = 0.0, 
                 p_erasure_check_error = 0.0):
        
        # Validate all probabilities
        for name, p in {
            "p_single_qubit_depolarize": p_single_qubit_depolarize,
            "p_two_qubit_depolarize": p_two_qubit_depolarize,
            "p_measurement_error": p_measurement_error,
            "p_single_qubit_erasure": p_single_qubit_erasure,
            "p_two_qubit_erasure": p_two_qubit_erasure,
            "p_erasure_check_error": p_erasure_check_error,
        }.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, got {p}")
        
        self.cpp_noise_model = _CPP_NoiseParams()
        self.cpp_noise_model.p_single_qubit_depolarize = p_single_qubit_depolarize
        self.cpp_noise_model.p_two_qubit_depolarize = p_two_qubit_depolarize
        self.cpp_noise_model.p_measurement_error = p_measurement_error
        self.cpp_noise_model.p_single_qubit_erasure = p_single_qubit_erasure
        self.cpp_noise_model.p_two_qubit_erasure = p_two_qubit_erasure
        self.cpp_noise_model.p_erasure_check_error = p_erasure_check_error
        # TODO : Cleaner way to do this?

    def __str__(self):
        return (f"NoiseModel(p_single_qubit_depolarize={self.p_single_qubit_depolarize}, "
                f"p_two_qubit_depolarize={self.p_two_qubit_depolarize}, "
                f"p_measurement_error={self.p_measurement_error}, "
                f"p_single_qubit_erasure={self.p_single_qubit_erasure}, "
                f"p_two_qubit_erasure={self.p_two_qubit_erasure}, "
                f"p_erasure_check_error={self.p_erasure_check_error})")
