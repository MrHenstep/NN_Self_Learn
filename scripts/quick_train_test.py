import time
import torch
from torch.utils.data import DataLoader
from cnns import CNN_load_datasets as ldd
from cnns import ResNet_model as rn

# Quick short training run to verify on-the-fly augmentation
random_seed = 0
torch.manual_seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

augment = True
bundle = ldd.load_dataset("cifar10", augment=augment)
train_ds, val_ds, test_ds = bundle.train, bundle.val, bundle.test

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4)

model = rn.ResNet(n_classes=10, use_projection=False).to(device)
criterion = torch.nn.CrossEntropyLoss()
# Exclude BatchNorm params and biases from weight decay
decay_params = []
no_decay_params = []
for name, p in model.named_parameters():
	if not p.requires_grad:
		continue
	n = name.lower()
	if n.endswith('.bias') or 'batchnorm' in n or ('.bn' in n) or ('bn' in n and 'batchnorm' not in n):
		no_decay_params.append(p)
	else:
		decay_params.append(p)

print(f"Quick test optimizer params: decay={len(decay_params)} no_decay={len(no_decay_params)}")

optimizer = torch.optim.SGD([
	{'params': decay_params, 'weight_decay': 1e-4},
	{'params': no_decay_params, 'weight_decay': 0.0}
], lr=0.1, momentum=0.9)
# short schedule for quick test
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

from cnns.CNN_train import train_epochs

start = time.time()
history = train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=2)
end = time.time()
print('History:')
print(history)
print('Elapsed total:', end - start)

# Run a quick validation test accuracy
from cnns.CNN_train import test_model

test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
print('Running test...')
test_model(model, test_loader, device)
