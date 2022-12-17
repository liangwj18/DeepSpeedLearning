import torch
import argparse
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


INTERFACE = 'ens6f0'
def get_network_info():
    ifstat = open('/proc/net/dev').readlines()
    for interface in ifstat:
        if INTERFACE in interface:
            receive = float(interface.split()[1])
            transmit = float(interface.split()[9])
            return receive, transmit

def print_info(content):
    if local_rank == 0:
        print(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST")
    parser.add_argument("--bsz", type=str, default=64)
    args = parser.parse_args()

    set_seed(42)
    DEVICE = "cuda"
    NUM_EPOCH = 5
    BATCH_SIZE = int(args.bsz)

    # Wrap the model
    model = ddp(ConvNet().to(DEVICE))
    node_size = flow.env.get_node_size()
    world_size = flow.env.get_world_size()
    rank = flow.env.get_rank()
    local_rank = flow.env.get_local_rank()

    print(f"node_size: {node_size}, world_size: {world_size}, rank: {rank}, local_rank: {local_rank}, bsz: {BATCH_SIZE}")

    training_data = datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
        # source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",  # 30000
    )
    train_sampler = flow.utils.data.distributed.DistributedSampler(
        training_data,
        num_replicas=world_size,
        rank=rank,
    )
    train_dataloader = flow.utils.data.DataLoader(
        training_data, BATCH_SIZE, shuffle=False, sampler=train_sampler,
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        download=True,
        # source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",
    )
    test_sampler = flow.utils.data.distributed.DistributedSampler(
        test_data,
        num_replicas=world_size,
        rank=rank,
    )
    test_dataloader = flow.utils.data.DataLoader(
        test_data, BATCH_SIZE, shuffle=False
    )
    print("training samples:", len(training_data))
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

    beg_receive, beg_transmit = get_network_info()

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

    end_receive, end_transmit = get_network_info()
    print(f"Total receive: {round((end_receive-beg_receive) / 1024 / 1024, 3)}MB, Total transmit: {round((end_transmit-beg_transmit) / 1024 / 1024, 3)}MB")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images.to(DEVICE))
            _, predicted = flow.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %%'.format(100 * correct / total))