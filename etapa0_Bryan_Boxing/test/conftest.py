# tests/conftest.py — v1.0
# Configuración global de pytest para la Etapa 0.
# Fija semillas globales antes de cualquier test para garantizar [NFR-4].

import random
import numpy as np
import torch


def pytest_configure(config):
    """Fija semillas globales al iniciar la sesión de pytest."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)