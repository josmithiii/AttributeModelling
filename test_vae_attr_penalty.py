import torch
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
    
    # Create a trainer with attribute regularization
    trainer = VAETrainer(
        dataset=dataset,
        model=model,
        lr=1e-4,
        has_reg_loss=True,  # Enable regularization
        reg_type='rhy_complexity',  # Target rhythmic complexity
        reg_dim=0  # Align with dimension 0
    )
    
    print(f"Trainer configured with regularization: {trainer.reg_type} aligned to dimension {trainer.reg_dim}")
    
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
    batch_data = (
        to_cuda_variable_long(score_tensor) if torch.cuda.is_available() else torch.tensor(score_tensor, dtype=torch.long),
        to_cuda_variable_long(metadata_tensor) if torch.cuda.is_available() else torch.tensor(metadata_tensor, dtype=torch.long)
    )
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    
    # Test the loss calculation including regularization
    print("Computing loss with regularization...")
    loss, accuracy = trainer.loss_and_acc_for_batch(batch_data, epoch_num=1, train=True)
    
    print(f"Loss with regularization: {loss.item()}")
    print(f"Accuracy: {accuracy}")
    
    # Compare with no regularization
    trainer_no_reg = VAETrainer(
        dataset=dataset,
        model=model,
        lr=1e-4,
        has_reg_loss=False
    )
    
    print("Computing loss without regularization...")
    loss_no_reg, accuracy_no_reg = trainer_no_reg.loss_and_acc_for_batch(batch_data, epoch_num=1, train=True)
    
    print(f"Loss without regularization: {loss_no_reg.item()}")
    print(f"Difference due to regularization: {loss.item() - loss_no_reg.item()}")
    
    print("Test complete!")