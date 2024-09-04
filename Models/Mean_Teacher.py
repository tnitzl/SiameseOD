import torch
import copy
import numpy as np
from Siamese.Losses.semi_supervised_loss import Semi_Supervised_Loss
from Siamese.Datasets.Datasets import Simple_Dataset


class Mean_Teacher(torch.nn.Module):
    def __init__(
        self,
        latend_dim: int,
        random_seed: int = 53,
        device: str = "cpu",
        lr: float = 0.001,
        epochs: int = 200,
        ema_decay: float = 0.99,
    ):
        super(Mean_Teacher, self).__init__()

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.activation = torch.nn.ReLU()

        # Student und teacher Models
        self.student = torch.nn.Sequential(
            torch.nn.Linear(latend_dim, 128),
            torch.nn.Dropout(0.3),
            self.activation,
            torch.nn.Linear(128, 128),
            torch.nn.Dropout(0.3),
            self.activation,
            torch.nn.Linear(128, 128),
            torch.nn.Dropout(0.3),
            self.activation,
            torch.nn.Linear(128, 1),
        )
        self.teacher = copy.deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.ema_decay: float = ema_decay

        # Config training
        self.device = device
        self.lr = lr
        self.epochs = epochs

        # save losses for prints
        self.epoch_losses = []
        self.supervised_losses = []
        self.unsupervised_losses = []

    def update_teacher(self) -> None:
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_param.data.mul_(self.ema_decay).add_(
                student_param.data, alpha=1 - self.ema_decay
            )

    def forward(self, x) -> list[torch.Tensor, torch.Tensor]:
        student = self.student(x)
        with torch.no_grad():
            teacher = self.teacher(x)
        return student, teacher

    def fit(self, x: np.ndarray, y: np.ndarray):

        self.train_loader, self.criterion = self.training_prepare(x, y)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.student.train()
        self.student.to(self.device)
        self.teacher.to(self.device)

        for epoch in range(self.epochs):
            epoch_loss: float = 0.0
            epoch_supervised_loss: float = 0.0
            epoch_unsupervised_loss: float = 0.0
            for inp, targets in self.train_loader:
                targets = targets.unsqueeze(1)
                inp, targets = (
                    inp.to(self.device),
                    targets.to(self.device),
                )
                student, teacher = self.forward(inp)

                self.optimizer.zero_grad()

                loss, supervised_loss, unsuperviesed_loss = self.criterion(
                    student, teacher, targets, 1
                )
                loss.backward()
                # Update teacher model
                self.update_teacher()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_supervised_loss += supervised_loss.item()
                epoch_unsupervised_loss += unsuperviesed_loss.item()
            self.epoch_losses.append(epoch_loss / len(self.train_loader))
            self.supervised_losses.append(
                epoch_supervised_loss / len(self.train_loader)
            )
            self.unsupervised_losses.append(
                epoch_unsupervised_loss / len(self.train_loader)
            )
            print(
                f"Epoche: [{epoch}/{self.epochs}], Loss: {epoch_loss/len(self.train_loader)}, Supervised_loss: {epoch_supervised_loss / len(self.train_loader)}, Unsupervised_loss: {epoch_unsupervised_loss / len(self.train_loader)} "
            )

        return self

    def training_prepare(self, x, y):
        simple_dataset: Simple_Dataset = Simple_Dataset(x, y)
        training_loader = torch.utils.data.DataLoader(
            dataset=simple_dataset, batch_size=256, shuffle=True
        )

        criterion = Semi_Supervised_Loss()
        return training_loader, criterion

    def predict(self, x: np.ndarray, y: np.ndarray):

        dataset = Simple_Dataset(x, y)
        predict_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=256, shuffle=True
        )

        y_pred = []
        y_true = []
        for emb, label in predict_loader:
            label = label.unsqueeze(1)
            emb = emb.to(self.device)
            _, out = self.forward(emb)
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
