import numpy as np

def save_mlp_npz(path: str, params: list, n_features: int, classes: np.ndarray, hidden_units: int):
    """
    Save sklearn MLP weights/biases + metadata to NPZ.
    params = [W0, W1, ..., b0, b1, ...] in your existing get_params/set_params format.
    """
    n_layers = len(params) // 2
    data = {
        "n_features": np.array([n_features], dtype=np.int32),
        "hidden_units": np.array([hidden_units], dtype=np.int32),
        "classes": classes.astype(np.int32),
        "n_layers": np.array([n_layers], dtype=np.int32),
    }
    # weights
    for i in range(n_layers):
        data[f"W{i}"] = params[i]
    # biases
    for i in range(n_layers):
        data[f"b{i}"] = params[n_layers + i]
    np.savez(path, **data)

def load_mlp_npz(path: str):
    """
    Load NPZ and return (params, n_features, classes, hidden_units)
    """
    z = np.load(path, allow_pickle=True)
    n_features = int(z["n_features"][0])
    hidden_units = int(z["hidden_units"][0])
    classes = z["classes"].astype(int)
    n_layers = int(z["n_layers"][0])

    params = []
    for i in range(n_layers):
        params.append(z[f"W{i}"])
    for i in range(n_layers):
        params.append(z[f"b{i}"])
    return params, n_features, classes, hidden_units
