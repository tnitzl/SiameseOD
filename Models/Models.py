import numpy as np
import torch
import copy

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Siamese.Datasets.Datasets import Base_Dataset, Siamese_Dataset, Simple_Dataset, Random_Dataset


class Anomaly_Detection_Model:
    def __init__(self, stacked_classifier: bool, device: str) -> None:

        # list of Siamese Model
        self.siamese_models: list[Siamese_Network] = []
        # lst of classifier_models
        self.classifier_models: list[Classifier_Network] = []

        # base dataset with all functions for the individuel datasets
        self.stacked_classifier: bool = stacked_classifier
        self.latend_dim: int = 50

        self.losses_siamese = []
        self.device = device

        # define datasets
        self.base_dataset: Base_Dataset = None
        self.random_dataset: Random_Dataset = None
        self.siamese_dataset: Siamese_Dataset = None
        self.unlabeld_dataset: Simple_Dataset = None
        self.full_dataset: Simple_Dataset = None
        self.labled_dataset: Simple_Dataset = None

        # define all loader classes
        self.siamese_loader: DataLoader = None
        self.classifier_loader: DataLoader = None

        self.embedded_loader: DataLoader = None

        # criterion und optimiuer
        self.criterion_siamese = None
        self.criterion_classifier = None
        self.optimizer_siamese = None
        self.optimizer_classifier = None

        # define input size
        self.input_size = None

    def prepare_base_dataset(
        self, x: np.ndarray, y: np.ndarray, percentage_labeld: float, len_dataset: int, siamese_model: str, classifier_model: str
    ) -> None:
        # Models müssen genauer angeschaut werden
        _, input_feature = x.shape()
        self.input_size = input_feature
        self.base_dataset = Base_Dataset(x, y, percentage_labeld)
        self.random_dataset = Random_Dataset(x, y, percentage_labeld, len_dataset)
        self.siamese_dataset = self.base_dataset.get_siamese_dataset()
        self.unlabeld_dataset = self.random_dataset.get_unlabeld_dataset()
        self.labled_dataset = self.random_dataset.get_labeld_dataset()
        self.full_dataset = self.random_dataset.get_complete_dataset()

        if siamese_model == 'random':
            self.siamese_loader = DataLoader(dataset = self.random_dataset, batch_size = 256)
        else:
            self.siamese_loader = DataLoader(dataset = self.siamese_dataset, batch_size = 256)

        if classifier_model == 'phi_model' or classifier_model == 'mean_teacher':
            self.classifier_loader = DataLoader(dataset = self.full_dataset, batch_size = 256, shuffle = True)
        else:
            self.classifier_loader = DataLoader(datset = self.labled_dataset, batch_size = 256, shuffle = True)


    def prepare_models(self, ensembles, stacked_classifier) -> None:
        for _ in range(len(ensembles)):
            model = Siamese_Network(
                self.base_dataset.input_size, self.latend_dim, device=self.device
            )
            self.siamese_models.append(model)
        if stacked_classifier:
            print(f'Das funktioniert leider nicht!')
            classifier = Classifier_Network(self.latend_dim * ensembles, self.device)
            self.classifier_models.append(classifier)
        else:
            for _ in range(len(ensembles)):
                classifier = Classifier_Network(self.latend_dim, self.device)
                self.classifier_model.append(classifier)


    def fit(self, **trainings_params) -> None:

        training_siamese = trainings_params.get("training_siamese")
        training_classifier = trainings_params.get("training_classifier")

        # Train Siamese models for every ensamble
        for siamese in self.siamese_models:
            epoch_losses = []
            for epoch in range(training_siamese["epochs"]):
                if self.verbose > 0:
                    print(
                        f"-------------------Start training Siamese Epoch {epoch}---------------"
                    )
                epoch_loss: float = 0.0
                for data in self.loader_siamese:
                    loss = siamese.train_forward(data)
                    epoch_loss += loss
                epoch_losses.append(epoch_loss)
                print(
                    f"Epoche [{epoch}/{len(training_siamese['epochs'])}], Loss: {epoch_loss}"
                )
                if self.verbose > 0:

                    print(
                        f"-------------------Finished training Siamese Epoch {epoch}---------------"
                    )
            self.losses_siamese.append(epoch_losses)

        if self.verbose > 0:
            print(f"Finished training for all Siamese Models")
        # train classifier model
        for siamese, classifier in zip(self.siamese_models, self.classifier_models):
            self.embedded_loader = self.create_embedding_loader(siamese)
            epoch_losses = []
            for epoch in range(training_classifier["epochs"]):
                if self.verbose > 0:
                    print(
                        f"-------------------Start training Classifier Epoch {epoch}---------------"
                    )
                epoch_loss: float = 0.0
                for data in self.embedded_loader:
                    loss = classifier.train_forward(data, self.classifier_criterium, self.classifier_optimizer)
                    epoch_loss += loss
                epoch_losses.append(epoch_loss)
                print(
                    f"Epoche [{epoch}/{len(training_siamese['epochs'])}], Loss: {epoch_loss}"
                )
                if self.verbose > 0:

                    print(
                        f"-------------------Finished training Siamese Epoch {epoch}---------------"
                    )
            self.losses_siamese.append(epoch_losses)

        if self.verbose > 0:
            print(f"Finished training for all Siamese Models")

    def create_embedding_loader(self, model):
        embeded_labeld_data = []
        embeded_labled_label = []
        
        for inp, label in self.classifier_loader:
            inp = inp.to(self.device)
            out, _ = model(inp, inp)
            out = out.to("cpu").detach().numpy()
            label = label.to("cpu").detach().numpy()
            embeded_labeld_data.extend(out)
            embeded_labled_label.extend(label)

        print(f'Die Länge der gelabelten embedded Daten ist {len(embeded_labeld_data)}')

        embeded_dataset: Simple_Dataset = Simple_Dataset(np.array(embeded_labeld_data), np.array(embeded_labled_label))
        loader_label_embedded = DataLoader(
            dataset = embeded_dataset,
            batch_size = 256,
            shuffle = True,
        )
        return loader_label_embedded


class Siamese_Network(nn.Module):
    def __init__(self, input_size: int, latend_dim: int, device: str) -> None:
        super(Siamese_Network, self).__init__()
        self.device = device

        # old settings
        # self.fc1 = nn.Linear(input_size, 20)
        # self.fc2 = nn.Linear(20, 32)
        # new settings
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, latend_dim)

    def forward_once(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        #x = F.selu(x)
        #x = F.leaky_relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def forward(self, x, y) -> (torch.Tensor, torch.Tensor):
        out1: torch.Tensor = self.forward_once(x)
        out2: torch.Tensor = self.forward_once(y)
        return out1, out2

    def training_forward(self, data, criterion, optimizer):
        self.train()
        optimizer.zero_grad()
        inp1, inp2, targets = data
        inp1, inp2, targets = (
            inp1.to(self.device),
            inp2.to(self.device),
            targets.to(self.device),
        )
        out1, out2 = self.forward(inp1, inp2)
        loss = criterion(out1, out2, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


class Classifier_Network(nn.Module):
    def __init__(self, latend_dim: int, device: str) -> None:
        super(Classifier_Network, self).__init__()
        self.device = device
        ## old seetings
        # self.fc1 = nn.Linear(32, 10)
        # self.fc2 = nn.Linear(10, 1)
        ## new seetings
        self.fc1 = nn.Linear(latend_dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x) -> torch.Tensor:
        #x = F.relu(self.fc1(x))
        #x = F.selu(self.fc1(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

    def training_forward(self, data, criterion, optimizer):
        self.train()
        optimizer.zero_grad()
        inp, targets = data
        inp, targets = inp.to(self.device), targets.to(self.device)
        out = self.forward(inp)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

'''
# https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
class ContrastiveLoss(nn.Module):
    # weighted loss
    def __init__(self, margin=20.0, verbose=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # for debugging
        self.verbose = verbose

    def forward(self, output1, output2, label):

        # euclidean_distance = torch.sqrt((output1 - output2) ** 2)

        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        if self.verbose >= 1:
            print(
                f"distance between the two vektores {output1}-{output2}: {euclidean_distance}"
            )
        # negativ_differenz = self.margin ** 2 - euclidean_distance ** 2
        # if self.verbose >= 1:
        #    print(negativ_differenz)

        bound_negativ_pair = (
            torch.max(self.margin**2 - euclidean_distance, torch.tensor(0.0)) ** 2
        )
        if self.verbose >= 1:
            print(f"bound negative pair:{bound_negativ_pair}")

        # https://dvl.in.tum.de/slides/adl4cv-ws20/2.Siamese.pdf
        loss_contrastive = (
            label * euclidean_distance**2 + (1 - label) * bound_negativ_pair
        )
        if self.verbose >= 1:
            print(f"contrastive loss: {loss_contrastive}")
        return loss_contrastive.mean()
'''


class Phi_Model(nn.Module):
    def __init__(self, latend_dim: int, device: str):
        super(Phi_Model, self).__init__()
        self.device: str = device

        self.layer1 = torch.nn.Linear(latend_dim, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def train_forward(self, data, criterion, optimizer):
        self.train()
        optimizer.zero_grad()
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        student = self.forward(inputs)
        teacher = self.forward(inputs)
        # Hier müssten eins studen teacher output sein. Noch einfügen, wenn genau klar ist wie die loss ist
        loss = criterion(student, teacher, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


def training_phi_step(model, data, criterion, optimizer):
    model.train()
    model = model.to("cpu")
    optimizer.zero_grad()

    # Daten trennen in gelabelt und ungelabelt
    inputs, targets = data
    targets = targets.long()
    # inputs, targets = inputs.to('cpu'), targets.to('cpu')
    labeled_mask = targets != -1
    unlabeled_mask = targets == -1

    inputs_labeled = inputs[labeled_mask]
    targets_labeled = targets[labeled_mask]
    inputs_unlabeled = inputs[unlabeled_mask]

    # Labeled data
    if len(inputs_labeled) > 0:
        outputs_labeled = model(inputs_labeled)
        supervised_loss = criterion(outputs_labeled, targets_labeled)
    else:
        supervised_loss = 0

    # Unlabeled data
    if len(inputs_unlabeled) > 0:
        # Zwei Vorhersagen mit verschiedenen Störungen
        outputs_unlabeled_1 = model(
            inputs_unlabeled + torch.randn_like(inputs_unlabeled) * 0.1
        )
        outputs_unlabeled_2 = model(
            inputs_unlabeled + torch.randn_like(inputs_unlabeled) * 0.1
        )
        consistency_loss = torch.nn.MSELoss()(outputs_unlabeled_1, outputs_unlabeled_2)
    else:
        consistency_loss = 0

    # Total loss
    total_loss = supervised_loss + consistency_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MeanTeacherModel(nn.Module):
    def __init__(self, model, ema_decay=0.99):
        super(MeanTeacherModel, self).__init__()
        self.student = model
        self.teacher = copy.deepcopy(model)
        self.ema_decay = ema_decay

        # Deaktiviere Gradienten für Teacher-Netzwerk
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self):
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_param.data.mul_(self.ema_decay).add_(
                student_param.data, alpha=1 - self.ema_decay
            )

    def forward(self, x):
        student_output = self.student(x)
        with torch.no_grad():
            teacher_output = self.teacher(x)
        return student_output, teacher_output

    def train_forward(self, data, criterion, optimizer):
        self.train()
        optimizer.zero_grad()
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        student, teacher = self.forward(inputs)
        # Hier müssten eins studen teacher output sein. Noch einfügen, wenn genau klar ist wie die loss ist
        loss = criterion(student, teacher, targets)
        loss.backward()
        optimizer.step()
        self.update_teacher()
        return loss.item()


def training_mean_teacher_step(model, data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    # Daten trennen in gelabelt und ungelabelt
    inputs, targets = data
    labeled_mask = targets != -1
    unlabeled_mask = targets == -1

    inputs_labeled = inputs[labeled_mask]
    targets_labeled = targets[labeled_mask]
    inputs_unlabeled = inputs[unlabeled_mask]

    # Labeled data
    if len(inputs_labeled) > 0:
        student_output_labeled, teacher_output_labeled = model(inputs_labeled)
        supervised_loss = criterion(student_output_labeled, targets_labeled)
    else:
        supervised_loss = 0

    # Unlabeled data
    if len(inputs_unlabeled) > 0:
        student_output_unlabeled, teacher_output_unlabeled = model(inputs_unlabeled)
        consistency_loss = nn.MSELoss()(
            student_output_unlabeled, teacher_output_unlabeled
        )
    else:
        consistency_loss = 0

    # Total loss
    total_loss = supervised_loss + consistency_loss
    total_loss.backward()
    optimizer.step()
    model.update_teacher()

    return total_loss.item()

def main() -> None:
    device: str = "mps"
    model = Anomaly_Detection_Model(False, device=device)
    first_example = {"test1": "test1", "test2": 100}
    second_example = {"second_test": "second", "second_test2": 20}
    model.fit(training_siamese=first_example, training_classifier=second_example)


if __name__ == "__main__":
    main()
