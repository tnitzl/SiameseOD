import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from Siamese.Datasets.Datasets import Base_Dataset, Siamese_Dataset, Simple_Dataset


class Classifier_Network(torch.nn.Module):
    def __init__(
        self, latend_dim: int, random_seed:int = 53, device: str = "cpu", lr: float = 0.001, epochs: int = 40
    ):
        super(Classifier_Network, self).__init__()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.fc1 = torch.nn.Linear(latend_dim, 10)
        self.fc2 = torch.nn.Linear(10, 1)

        # Activationfunction
        self.activation = torch.nn.functional.relu

        # Config training
        self.device = device
        self.lr = lr
        self.epochs = epochs

        # save_losses for prints
        self.epoch_losses = []

        print(f'Model initialisiert mir random_seed = {random_seed}')

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = torch.nn.functional.sigmoid(x)
        return x

    def fit(self, x: np.ndarray, y: np.ndarray):

        self.train_loader, self.criterion = self.training_prepare(x, y)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # self.train()

        for epoch in range(self.epochs):
            epoch_loss: float = 0.0
            for inp, targets in self.train_loader:
                targets = targets.unsqueeze(1)
                inp, targets = (
                    inp.to(self.device),
                    targets.to(self.device),
                )
                out = self.forward(inp)

                self.optimizer.zero_grad()

                loss = self.criterion(out, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.epoch_losses.append(epoch_loss / len(self.train_loader))
            print(
                f"Epoche: [{epoch}/{self.epochs}], Loss: {epoch_loss/len(self.train_loader)} "
            )

        return self

    def training_prepare(self, x, y):
        simple_dataset: Simple_Dataset = Simple_Dataset(np.array(x), np.array(y))
        training_loader = torch.utils.data.DataLoader(
            dataset=simple_dataset, batch_size=256, shuffle=True
        )

        citerion = torch.nn.functional.binary_cross_entropy
        return training_loader, citerion

    def predict(self, x: np.ndarray, y: np.ndarray):

        dataset = Simple_Dataset(x, y)
        predict_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=256
        )

        y_pred = []
        y_true = []
        for emb, label in predict_loader:
            label = label.unsqueeze(1)
            emb = emb.to(self.device)
            out = self.forward(emb)
            out = out.to("cpu").detach().numpy()
            label = label.to("cpu").detach().numpy()
            label = label.reshape(-1)
            out = out.reshape(-1)
            # y_pred.extend(np.argmax(out, axis=1))
            y_pred.extend(out)
            y_true.extend(label)

        # roc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        # pr = average_precision_score(y_true, y_pred)
        return y_pred, y_true
