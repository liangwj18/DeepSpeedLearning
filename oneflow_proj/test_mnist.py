import torch
import oneflow as flow
from datetime import datetime
import oneflow.nn as nn
import oneflow.optim as optim
from flowvision import datasets
from flowvision import transforms
from oneflow.nn.parallel import DistributedDataParallel as ddp
from utils import set_seed


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# def train(iter, model, loss_fn, optimizer):
#     size = len(iter.dataset)
#     for batch, (x, y) in enumerate(iter):
#         x = x.to(DEVICE)
#         y = y.to(DEVICE)
#
#         # Compute prediction error
#         pred = model(x)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         current = batch * BATCH_SIZE
#         if batch % 100 == 0:
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test(iter, model, loss_fn):
#     size = len(iter.dataset)
#     num_batches = len(iter)
#     model.eval()
#     test_loss, correct = 0, 0
#     with flow.no_grad():
#         for x, y in iter:
#             x = x.to(DEVICE)
#             y = y.to(DEVICE)
#
#             pred = model(x)
#             test_loss += loss_fn(pred, y)
#             bool_value = (pred.argmax(1).to(dtype=flow.int64) == y)
#             correct += float(bool_value.sum().numpy())
#     test_loss /= num_batches
#     print("test_loss", test_loss, "num_batches ", num_batches)
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}, Avg loss: {test_loss:>8f}")


def print_info(content):
    if local_rank == 0:
        print(content)


if __name__ == '__main__':
    set_seed(42)
    BATCH_SIZE = 100
    NUM_EPOCH = 5
    DEVICE = "cuda"

    # Wrap the model
    model = ddp(ConvNet().to(DEVICE))
    node_size = flow.env.get_node_size()
    world_size = flow.env.get_world_size()
    rank = flow.env.get_rank()
    local_rank = flow.env.get_local_rank()

    print(f"node_size: {node_size}, world_size: {world_size}, rank: {rank}, local_rank: {local_rank}")
    # print(f"rank: {rank}")

    # Data loading code
    # train_dataset = torchvision.datasets.MNIST(
    #     root='./data',
    #     train=True,
    #     transform=transforms.ToTensor(),
    #     download=True
    # )
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset,
    #     num_replicas=args.world_size,
    #     rank=rank
    # )
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=False,
    #                                            num_workers=0,
    #                                            pin_memory=True,
    #                                            sampler=train_sampler)

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
        source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",
    )
    # 30000

    train_sampler = flow.utils.data.distributed.DistributedSampler(
        training_data,
        num_replicas=world_size,
        rank=rank,
    )
    train_dataloader = flow.utils.data.DataLoader(
        training_data, BATCH_SIZE, shuffle=False, sampler=train_sampler,
    )
    # test_data = datasets.FashionMNIST(
    #     root="data",
    #     train=False,
    #     transform=transforms.ToTensor(),
    #     download=True,
    #     source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",
    # )
    # test_dataloader = flow.utils.data.DataLoader(
    #     test_data, BATCH_SIZE, shuffle=False
    # )
    for x, y in train_dataloader:
        print_info(f"x.shape: {x.shape}, y.shape: {y.shape}")
        break

    start = datetime.now()
    total_step = len(train_dataloader)
    print_info(f"total_step: {total_step}")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), 1e-4)

    # Data Parallel TODO
    # placement = flow.placement(type="cuda", ranks=[0, 1])

    for epoch in range(NUM_EPOCH):
        for i, (images, labels) in enumerate(train_dataloader):
            # images = images[rank].cuda()
            # labels = labels[rank].cuda()
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # print_info(f"x.shape: {images.shape}, y.shape: {labels.shape}")

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and local_rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, NUM_EPOCH, i + 1, total_step,loss.item()))
    print_info("Training complete in: " + str(datetime.now() - start))
