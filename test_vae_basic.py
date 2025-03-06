import torch
from AttributeModelling.MeasureVAE.measure_vae import MeasureVAE
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
    print(f"Metadata tensor shape: {metadata_tensor.shape}")

    # Test forward pass
    score_tensor = to_cuda_variable_long(score_tensor) if torch.cuda.is_available() else torch.tensor(score_tensor, dtype=torch.long)
    metadata_tensor = to_cuda_variable_long(metadata_tensor) if torch.cuda.is_available() else torch.tensor(metadata_tensor, dtype=torch.long)

    if torch.cuda.is_available():
        model.cuda()

    print("Running forward pass...")
    weights, samples, z_dist, prior_dist, z_tilde, z_prior = model(
        measure_score_tensor=score_tensor,
        measure_metadata_tensor=metadata_tensor,
        train=True
    )

    print("Forward pass successful!")
    print(f"Z distribution mean shape: {z_dist.loc.shape}")
    print(f"Output weights shape: {weights.shape}")
    print(f"Output samples shape: {samples.shape}")

    print("Test complete!")