from unicodedata import name
from qerasure_python import NoiseParams as _CPP_NoiseParams

class NoiseParams:
    """
    A Python wrapper around the C++ NoiseModel class.

    This class provides a Python interface to the C++ implementation of noise models for quantum error correction.
    It initializes the noise model with given parameters and exposes properties such as the error probabilities for different types of errors.
    """
    def __init__(self, **kwargs):
        self._cpp_noise_params = _CPP_NoiseParams()
        # Set the noise parameters based on the provided keyword arguments
        for err_mech, prob in kwargs.items():
            self._cpp_noise_params.set(err_mech, prob)

    def set(self, err_mech, prob):
        self._cpp_noise_params.set(err_mech, prob)

    def get(self, err_mech):
        return self._cpp_noise_params.get(err_mech)
    
    def __repr__(self):
        return self._cpp_noise_params.__repr__()
    
    # Allow pythonic access to the noise parameters as attributes
    def __getattr__(self, err_mech):
        return getattr(self._cpp_noise_params, err_mech)
    
    # Allow setting noise parameters as attributes
    def __setattr__(self, err_mech, prob):
        if err_mech == "_cpp_noise_params":
            super().__setattr__(err_mech, prob)
        else:
            setattr(self._cpp_noise_params, err_mech, prob)
