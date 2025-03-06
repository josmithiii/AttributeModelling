import torch
import torch.nn.functional as F
from torch import distributions
from AttributeModelling.MeasureVAE.measure_vae import MeasureVAE
from AttributeModelling.MeasureVAE.vae_trainer import VAETrainer
from AttributeModelling.data.dataloaders.bar_dataset import *
from AttributeModelling.utils.helpers import *

if __name__ == "__main__":
    print("Loading dataset...")
    is_short = True
    dataset = FolkBarDataset(dataset_type='train', is_short=is_short)

    print("Creating model...")
    model = MeasureVAE(
        dataset=dataset,
        note_embedding_dim=10,
        metadata_embedding_dim=2,
        num_encoder_layers=2,
        encoder_hidden_size=64,  # smaller for testing
        encoder_dropout_prob=0.5,
        latent_space_dim=32,  # smaller for testing
        num_decoder_layers=2,
        decoder_hidden_size=64,  # smaller for testing
        decoder_dropout_prob=0.5,
        has_metadata=False
    )

    print("Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create a small test batch
    (train_loader, val_loader, test_loader) = dataset.data_loaders(
        batch_size=2,
        split=(0.7, 0.2)
    )

    # Get a batch
    for batch_data in train_loader:
        score_tensor, metadata_tensor = batch_data
        break

    print(f"Score tensor shape: {score_tensor.shape}")
    
    # Convert to appropriate format
    score_tensor = to_cuda_variable_long(score_tensor) if torch.cuda.is_available() else torch.tensor(score_tensor, dtype=torch.long)
    metadata_tensor = to_cuda_variable_long(metadata_tensor) if torch.cuda.is_available() else torch.tensor(metadata_tensor, dtype=torch.long)
    batch_data = (score_tensor, metadata_tensor)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    
    # Let's manually run the model and compute the normal loss
    print("Running model forward pass...")
    weights, samples, z_dist, prior_dist, z_tilde, z_prior = model(
        measure_score_tensor=score_tensor,
        measure_metadata_tensor=metadata_tensor,
        train=True
    )
    
    # Compute regular loss without attribute penalty
    recons_loss = VAETrainer.mean_crossentropy_loss(weights=weights, targets=score_tensor)
    
    # Compute KL divergence
    kld = distributions.kl.kl_divergence(z_dist, prior_dist)
    beta = 0.001  # Same as in VAETrainer
    kld_loss = beta * kld.sum(1).mean()
    
    # Total loss without attribute penalty
    loss_no_penalty = recons_loss + kld_loss
    print(f"Loss without attribute penalty: {loss_no_penalty.item()}")
    
    # Now, let's manually compute the attribute penalty
    print("Computing attribute penalty...")
    
    # Simple attribute - note density
    attr_tensor = torch.sum((score_tensor > 0).float(), dim=1, keepdim=True) / score_tensor.size(1)
    
    # Use dimension 0 of latent space
    latent_dim = 0
    x = z_tilde[:, latent_dim]
    
    # Prepare data for sign loss 
    x_expanded = x.view(-1, 1).repeat(1, x.shape[0])
    x_diff_sign = (x_expanded - x_expanded.transpose(1, 0)).view(-1, 1)
    x_diff_sign = torch.tanh(x_diff_sign * 10)
    
    # Prepare labels
    y = attr_tensor.view(-1, 1).repeat(1, attr_tensor.shape[0])
    y_diff_sign = torch.sign(y - y.transpose(1, 0)).view(-1, 1)
    
    # Compute sign loss
    loss_fn = torch.nn.L1Loss()
    attr_penalty = loss_fn(x_diff_sign, y_diff_sign)
    
    print(f"Attribute penalty value: {attr_penalty.item()}")
    
    # Total loss with attribute penalty
    total_loss = loss_no_penalty + attr_penalty
    print(f"Total loss with attribute penalty: {total_loss.item()}")
    print(f"Difference due to attribute penalty: {attr_penalty.item()}")
    
    print("Test complete!")