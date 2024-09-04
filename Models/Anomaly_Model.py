import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score

from Siamese.Models.Siamese_model import Siamese_model
from Siamese.Models.Classifier import Classifier_Network


class Anomaly_Model:
    def __init__(
        self,
        input_size: int,
        random_seed: int = 53,
        latend_dim: int = 50,
        device: str = "cpu",
        embedded_model: Siamese_model = None,
        classifier_model: Classifier_Network = None,
    ):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.input_size = input_size
        self.latend_dim = latend_dim

        # Models
        self.embedded_model: Siamese_model = None
        if embedded_model == None:
            self.embedded_model = Siamese_model(
                self.input_size, self.latend_dim, device=device
            )
        else:
            self.embedded_model = embedded_model

        self.classifier_model = None
        if classifier_model == None:
            self.classifier_model = Classifier_Network(self.latend_dim, device=device)
        else:
            self.classifier_model = classifier_model

        print(self.embedded_model)
        print(self.classifier_model)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        percent_labeled: float,
        semi_supervised: bool = False,
    ):
        print(f"Variable semisupervised anomaly Model: {semi_supervised}")

        self.embedded_model.fit(x, y, percent_labeled)
        embedded_data, embedded_label = self.embedded_model.create_embedding_training(
            semi_supervised=semi_supervised
        )
        print(
            f"Das ist der typ der Daten die zurückkommen: {type(embedded_data[0])}, {type(embedded_label[0])}"
        )
        self.classifier_model.fit(embedded_data, embedded_label)

    # def training_prepare(self):
    #    pass

    def predict(self, x: np.ndarray, y: np.ndarray):
        data, label = self.embedded_model.create_embedding_prediction(x, y)
        y_pred, y_true = self.classifier_model.predict(data, label)

        roc = roc_auc_score(y_true, y_pred, multi_class="ovr")
        pr = average_precision_score(y_true, y_pred)
        print(f"Das sind die Ergebnisse: Roc {roc}, Pr {pr}")
        return y_pred, y_true, roc, pr
