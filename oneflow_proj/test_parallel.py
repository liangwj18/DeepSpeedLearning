import unittest
import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

class TestParallel(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_data_parallel(self):
        BATCH_SIZE = 64
        EPOCH_NUM = 1

        PLACEMENT = flow.placement("cuda", [0, 1])
        S0 = flow.sbp.split(0)
        B = flow.sbp.broadcast

        DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
        print("Using {} device".format(DEVICE))

        training_data = flowvision.datasets.CIFAR10(
            root="data",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )

        train_dataloader = flow.utils.data.DataLoader(
            training_data, BATCH_SIZE, shuffle=True
        )

        model = flowvision.models.mobilenet_v2().to(DEVICE)
        model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
        model = model.to_global(placement=PLACEMENT, sbp=B)

        loss_fn = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

        for t in range(EPOCH_NUM):
            print(f"Epoch {t + 1}\n-------------------------------")
            size = len(train_dataloader.dataset)
            for batch, (x, y) in enumerate(train_dataloader):
                x = x.to_global(placement=PLACEMENT, sbp=S0)
                y = y.to_global(placement=PLACEMENT, sbp=S0)

                # Compute prediction error
                pred = model(x)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current = batch * BATCH_SIZE
                if batch % 5 == 0:
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':
    TestParallel().run()