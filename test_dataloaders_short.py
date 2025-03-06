from AttributeModelling.data.dataloaders.bar_dataset import*

# Use a very small dataset for testing
is_short = True
num_bars = 1
batch_size = 2
print("Loading dataset...")
bar_dataset = FolkBarDataset(dataset_type='train', is_short=is_short)
(train_dataloader,
 val_dataloader,
 test_dataloader) = bar_dataset.data_loaders(
    batch_size=batch_size,
    split=(0.7, 0.2)
)
print('Num Train Batches: ', len(train_dataloader))
print('Num Valid Batches: ', len(val_dataloader))
print('Num Test Batches: ', len(test_dataloader))