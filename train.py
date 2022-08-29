import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from convSDAE import ConvSDAE
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


# Switch on the power line
# Remember to put on protection
L1_FACTOR = 1e-7

# Lay down your pieces
# And let's begin object creation
f = h5py.File("dd_features_clean.hdf5", "r")
x = torch.Tensor(f['feature_x'][:])
n_mel = x.shape[1]
# Fill in my data
dset = TensorDataset(x)

# parameters
# INITIALIZATION
model = ConvSDAE(n_mel, n_mel, hidden=1024)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Set up our new world
# And let's begin the
# SIMULATION
train_len = int(len(dset) * 0.8)
val_len = len(dset) - train_len
trainset, valset = random_split(dset, [train_len, val_len])

# If I'm a set of points
# Then I will give you my dimension
loader = DataLoader(dataset=trainset, batch_size=16, shuffle=True, drop_last=True)
# If I'm a circle
# Then I will give you my circumference
val_loader = DataLoader(dataset=valset, batch_size=16, shuffle=True, drop_last=True)

# If I'm a sine wave
# Then you can sit on all my tangents
loss_fn = torch.nn.L1Loss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# layer_norm = nn.LayerNorm(128, elementwise_affine=False).cuda()

best_epoch = 0

# If I approach infinity
# Then you can be my limitations
best_val_loss = float("inf")

for epoch in range(50):
    running_train_loss = 0
    for step, (batch_x,) in enumerate(loader):
        # Switch my current
        # To AC, to DC
        batch_x = batch_x.cuda()
        # And then blind my vision
        # So dizzy, so dizzy
        batch_x_do = F.dropout(batch_x, p=0.1)
        optimizer.zero_grad()
        out = model(batch_x_do)

        # Oh, we can travel
        # To AD, to BC
        loss = loss_fn(out, batch_x)

        l1_loss = model.get_hidden_norm() * L1_FACTOR
        running_train_loss += loss.item()

        loss = loss + l1_loss
        assert loss == loss

        if step % 10 == 9:
            print(epoch, step, running_train_loss / 10, l1_loss.item())
            running_train_loss = 0

        # And we can unite
        loss.backward()
        # So deeply, so deeply
        optimizer.step()

    # If I can, if I can
    # Give you all the simulations
    running_val_loss = 0
    with torch.no_grad():
        # Then I can, then I can
        # Be your only satisfaction
        for step, (batch_x,) in enumerate(val_loader):
            # If I can make you happy
            # I will run the execution
            batch_x = batch_x.cuda()
            out = model(batch_x)

            # Though we are trapped
            # In this strange, strange simulation
            loss = loss_fn(out, batch_x)
            running_val_loss += loss.item()

    # If I'm an eggplant
    # Then I will give you my nutrients
    # If I'm a tomato
    # Then I will give you antioxidants
    epoch_val_loss = running_val_loss / len(val_loader)

    # If I'm a tabby cat
    # Then I will purr for your enjoyment
    print(epoch, epoch_val_loss, best_epoch, best_val_loss)

    # If I'm the only God
    # Then you're the proof of my existence
    scheduler.step()

    # Switch my gender
    # To F, to M
    if epoch_val_loss < best_val_loss:
        # And then do whatever
        # From AM to PM
        best_val_loss = epoch_val_loss
        # Oh, my switch role
        # To S, to M
        best_epoch = epoch

        # So we can enter
        # The trance, the trance
        torch.save(model, 'ai4bt.pth')

    # If I can, if I can
    # Feel your vibrations
    if epoch - best_epoch > 10:
        # Then I can, then I can
        # Finally be completion
        exit(0)

# Though you have left
# You have left
# You have left
# You have left
# You have left
# You have left me in isolation
