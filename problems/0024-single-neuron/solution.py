import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # 1. Tensors banana (Yeh tumne bilkul sahi kiya tha)
    ten_features = torch.tensor(features, dtype=torch.float32)
    ten_weights = torch.tensor(weights, dtype=torch.float32)
    ten_label = torch.tensor(labels, dtype=torch.float32)

    # 2. Linear combination aur Sigmoid activation
    z = torch.matmul(ten_features, ten_weights) + bias
    y_hat = torch.sigmoid(z) # Tensor hi rehne do

    # 3. MSE Loss calculate karo (Tensor ke sath)
    loss = F.mse_loss(y_hat, ten_label)

    # 4. Return format match karna:
    # y_hat ko list mein badal kar 4 decimals tak round karo
    y_hat_list = [round(val, 4) for val in y_hat.tolist()]
    
    # loss ek single number (scalar) hai, toh .item() se float nikalkar round karo
    loss_val = round(loss.item(), 4)

    return y_hat_list, loss_val