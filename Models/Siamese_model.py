import torch
import numpy as np
from Siamese.Datasets.Datasets import (
    Base_Dataset,
    Siamese_Dataset,
    Simple_Dataset,
    Random_Dataset,
)
from Siamese.Losses.contrastiv_loss import ContrastiveLoss_cosinesimularity

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Siamese_model(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        latend_dim: int,
        random_seed: int = 53,
        device: str = "cpu",
        lr: int = 0.0001,
        epochs: int = 20,
    ) -> None:
        super(Siamese_model, self).__init__()

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.fc1 = torch.nn.Linear(input_size, 100)
        self.fc2 = torch.nn.Linear(100, latend_dim)

        # Activationfunction
        self.activation = torch.nn.functional.relu

        # Config training
        self.lr = lr
        self.epochs = epochs
        self.device = device

        # save_losses for prints
        self.epoch_losses = []

        print(f'Siamese model initialisert mit random seed = {random_seed}')

    def forward_one(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def forward(self, x1, x2) -> [torch.Tensor, torch.Tensor]:
        out1: torch.Tensor = self.forward_one(x1)
        out2: torch.Tensor = self.forward_one(x2)
        return out1, out2

    # Funtion to train the model
    def fit(self, x: np.ndarray, y: np.ndarray, percent_labeled: float):

        self.train_loader, self.criterion = self.training_prepare(x, y, percent_labeled)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        self.to(self.device)
        self.train()

        for epoch in range(self.epochs):
            epoch_loss: float = 0.0
            for inp1, inp2, targets in self.train_loader:
                inp1, inp2, targets = (
                    inp1.to(self.device),
                    inp2.to(self.device),
                    targets.to(self.device),
                )

                out1, out2 = self.forward(inp1, inp2)

                self.optimizer.zero_grad()
                loss = self.criterion(out1, out2, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.epoch_losses.append(epoch_loss / len(self.train_loader))
            print(
                f"Epoche: [{epoch}/{self.epochs}], Loss: {epoch_loss/len(self.train_loader)} "
            )

        return self

    # prepare Training
    def training_prepare(self, x, y, percentage_labeld):
        base_dataset: Base_Dataset = Base_Dataset(x, y, percentage_labeld)
        # base_dataset: Random_Dataset = Random_Dataset(x, y, percentage_labeld)
        siames_dataset: Siamese_Dataset = base_dataset.get_siamese_dataset()
        # siames_dataset = base_dataset
        self.unlabeld_dataset: Simple_Dataset = base_dataset.get_unlabeld_dataset()
        self.labled_dataset: Simple_Dataset = base_dataset.get_labeld_dataset()
        self.complete_dataset: Simple_Dataset = base_dataset.get_complete_dataset()

        training_loader = torch.utils.data.DataLoader(
            dataset=siames_dataset, batch_size=256, shuffle=True
        )

        citerion: ContrastiveLoss_cosinesimularity = ContrastiveLoss_cosinesimularity(
            margin=1.0
        )
        return training_loader, citerion

    def create_embedding_training(
        self, x: np.ndarray = None, y: np.ndarray = None, semi_supervised: bool = False
    ) -> list[list[int], list[np.float32]]:
        print(f"Semispvervised variable siamese model {semi_supervised}")
        embedded_loader = None
        if semi_supervised:
            print(f"Semisupervised dataset wird initialisiert!")
            embedded_loader = torch.utils.data.DataLoader(
                dataset=self.complete_dataset,
                batch_size=256,
            )
        elif x == None and y == None:
            embedded_loader = torch.utils.data.DataLoader(
                dataset=self.labled_dataset, batch_size=256
            )
        else:
            dataset = Simple_Dataset(x, y)
            embedded_loader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=256, shuffle=True
            )

        self.embeded_labeld_data: list[int] = []
        self.embeded_labled_label: list[int] = []
        for inp, label in embedded_loader:
            inp = inp.to(self.device)
            out, _ = self.forward(inp, inp)
            out = out.to("cpu").detach().numpy()
            label = label.to("cpu").detach().numpy()
            self.embeded_labeld_data.extend(out)
            self.embeded_labled_label.extend(label)

        return self.embeded_labeld_data, self.embeded_labled_label

    def create_embedding_prediction(self, x: np.ndarray = None, y: np.ndarray = None):
        embedded_loader = torch.utils.data.DataLoader(
            dataset=self.unlabeld_dataset, batch_size=256, shuffle=True
        )
        # embedded_loader = None
        # if x == None and y == None:
        #    embedded_loader = torch.utils.data.DataLoader(
        #        dataset = self.unlabeld_dataset,
        #        batch_size = 256,
        #        shuffle = True
        #    )
        # else:
        #    dataset = Simple_Dataset(x, y)
        #    embedded_loader = torch.utils.data.DataLoader(
        #        dataset=dataset, batch_size=256, shuffle=True
        #    )

        self.embeded_unlabeld_data = []
        self.embeded_unlabled_label = []
        for inp, label in embedded_loader:
            inp = inp.to(self.device)
            out = self.forward_one(inp)
            out = out.to("cpu").detach().numpy()
            label = label.to("cpu").detach().numpy()
            self.embeded_unlabeld_data.extend(out)
            self.embeded_unlabled_label.extend(label)

        return self.embeded_unlabeld_data, self.embeded_unlabled_label

    def print_embeddings(self, training: bool = True):
        gelabeld: np.ndarray = None
        lables: np.ndarray = None

        if training:
            gelabeld = np.array(self.embeded_labeld_data)
            lables = np.array(self.embeded_labled_label)
        else:
            gelabeld = np.array(self.embeded_unlabeld_data)
            lables = np.array(self.embeded_unlabled_label)

        if len(gelabeld) >= 30:
            print(f"plot embedded gelabeldete Daten")
            tnse = TSNE(n_components=2, random_state=42)

            features_2d = tnse.fit_transform(gelabeld)

            plt.figure(figsize=(10, 7))
            scatter = plt.scatter(
                features_2d[:, 0],
                features_2d[:, 1],
                c=lables,
                cmap="viridis",
                alpha=0.6,
            )
            plt.colorbar(scatter)
            plt.title("t-SNE Visualisierung der 32-dimensionalen Vektoren")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.show()
