import torch


def dataloader_to_data(train_data):
    # TODO: check if there is a better way to use dataloaders with botorch
    train_x = torch.Tensor()
    train_y = torch.Tensor()
    for state, score in train_data:
        train_x = torch.cat((train_x, state), 0)
        train_y = torch.cat((train_y, score), 0)

    return train_x, train_y
