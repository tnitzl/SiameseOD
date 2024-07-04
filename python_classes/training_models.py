import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random

from Siamese.python_classes.Datasets import Base_Dataset, Simple_Dataset, Siamese_Dataset, Random_Dataset
from Siamese.python_classes.Models import Siamese_Network, Classifier_Network, ContrastiveLoss
from sklearn.metrics import roc_auc_score, average_precision_score

import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def train_dataset(file: str, 
                  random_seed: int, 
                  percentage_labeld:float = 0.05, 
                  contrastiv_margin: float = 5.0, 
                  lr_siamese: float = 0.0001, 
                  lr_classifier: float = 0.001, 
                  epochs_siamese: int = 20, 
                  epochs_classifier: int = 40, 
                  print_embeddeds: bool = False, 
                  print_learning: int =False
                ):

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    #device: str = "cpu"
    device: torch.device = torch.device(device)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    current_directory: str = os.getcwd() # Current directory ../Masterarbeit//Siames
    #parent_dirctory: str = os.path.dirname(current_directory)
    file_classic_dataset: str = "/Archiv/Classical/" # Folder with alle classic ADBench datasets
    file = file
    #file: str = "32_shuttle.npz" # check correct impelentation with the first dataset
    # later it will be a for loop with all .npz files

    print(f'##############Start Training with Dataset {file}######################')

    data: np.lib.npyio.NpzFile = np.load(current_directory + file_classic_dataset + file, allow_pickle = True)
    X: np.ndarray = data['X']
    y: np.ndarray = data['y']

    samples, input_features = X.shape

    base_dataset: Base_Dataset = Base_Dataset(X=X, y=y, percent_labeld=percentage_labeld)
    siamese_dataset: Siamese_Dataset = base_dataset.get_siamese_dataset()
    unlabeld_dataset: Simple_Dataset = base_dataset.get_unlabeld_dataset()
    labeld_dataset: Simple_Dataset = base_dataset.get_labeld_dataset()

    loader_siamese = DataLoader(
        dataset = siamese_dataset,
        batch_size = 256,
        shuffle = True
    )

    loader_unlabeld_data = DataLoader(
        dataset = unlabeld_dataset,
        batch_size = 256,
    )

    loader_label_data = DataLoader(
        dataset = labeld_dataset,
        batch_size = 256,
    )
    print(f'Die länge des ungelabendeten Datensatzen ist: {len(unlabeld_dataset)}')
    print(f'Die länge des ungelabendeten Datenloader ist: {len(loader_unlabeld_data)}')

    siamese_network: Siamese_Network = Siamese_Network(input_size = input_features).to(device)
    classfier_network: Classifier_Network = Classifier_Network().to(device)

    contrastiv_loss: ContrastiveLoss = ContrastiveLoss(margin = contrastiv_margin)
    mse_loss =  F.binary_cross_entropy
    print(f'Das ist die verwendete loss{mse_loss}')

    optimizer_siamese = torch.optim.Adam(siamese_network.parameters(), lr=lr_siamese)
    optimizer_classifier = torch.optim.Adam(classfier_network.parameters(), lr=lr_classifier)

    list_epoch_loss = [] 
    loss_iteration = []
    for epoch in range(epochs_siamese):
        print(f'----------------Start trainign Epoche {epoch}----------------')
        epoch_loss: float = 0.0
        #loss_10000_iteration: float = 0.0
        #counter = 0
        for (inp1, inp2, targets) in loader_siamese:
            inp1, inp2, targets = inp1.to(device), inp2.to(device), targets.to(device)
            output1, output2 = siamese_network(inp1, inp2)
            loss = contrastiv_loss(output1, output2, targets)
            loss.backward()
            optimizer_siamese.step()
            iteration_loss = loss.item()
            loss_iteration.append(iteration_loss)
            epoch_loss += loss.item()
            #loss_10000_iteration += loss.item()
            #if counter % 100 == 0:
            #    print(f'Die loss für 10000 trainingschleifen ist {loss_10000_iteration / 10000}')
            #    loss_10000_iteration = 0
            #    counter = 0
            #counter += 1
        list_epoch_loss.append(epoch_loss/len(loader_siamese))
        print(f'Epoche: {epoch} Average Loss: {epoch_loss/len(loader_siamese)}')
    print("---------Finished Training Siamese Network-------------")

    embeded_labeld_data = []
    embeded_labled_label = []
    for inp, label in loader_label_data:
        inp = inp.to(device)
        out, _ = siamese_network(inp, inp)
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
    
    if print_embeddeds:
        if len(embeded_labeld_data) >= 30:
            print(f'plot embedded gelabeldete Daten')
            tnse = TSNE(n_components=2, random_state=42)
    
            gelabeld = np.array(embeded_labeld_data)
            lables = np.array(embeded_labled_label)

            features_2d = tnse.fit_transform(gelabeld)

            plt.figure(figsize=(10,7))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=lables, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
            plt.title('t-SNE Visualisierung der 32-dimensionalen Vektoren')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.show()

    if print_learning:
        plt.plot(list_epoch_loss)
        plt.title('Loss Curve Siamese')
        plt.show()

    list_epoch_loss = [] 
    loss_iteration = []
    for epoch in range(epochs_classifier):
        print(f'----------------Start trainign Epoche {epoch}----------------')
        epoch_loss: float = 0.0
        for embed, lab in loader_label_embedded:
            lab = lab.unsqueeze(1)
            embed, lab = embed.to(device), lab.to(device)

            out = classfier_network(embed)

            loss = mse_loss(out, lab)
            loss.backward()
            optimizer_classifier.step()
            iteration_loss = loss.item()
            loss_iteration.append(iteration_loss)

            epoch_loss += iteration_loss

        list_epoch_loss.append(epoch_loss/len(loader_label_embedded))
        print(f'Epoche: {epoch} Average Loss: {epoch_loss/len(loader_label_embedded)}')
    
    print("------------------Finished Training Classifier----------------------")

    embeded_unlabeld_data = []
    embeded_unlabeld_label = []
    #counter = 0
    for (inp, label) in loader_unlabeld_data:
        inp = inp.to(device)
        out, _ = siamese_network(inp, inp)
        out = out.to("cpu").detach().numpy()
        label = label.to("cpu").detach().numpy()
        embeded_unlabeld_data.extend(out)
        embeded_unlabeld_label.extend(label)
        #print(f'Counter durchläufe loader unlabeld data {counter}')
        #counter += 1

    print(f'Das ist die länge der ungelabelten Daten nach dem Siamese {len(embeded_unlabeld_data)}')
    print(f'Das ist die länge der ungelabelten Labels nach dem Siamese {len(embeded_unlabeld_label)}')

    embeded_unalbeld_dataset: Simple_Dataset = Simple_Dataset(np.array(embeded_unlabeld_data), np.array(embeded_unlabeld_label))
    loader_unlabel_embedded = DataLoader(
        dataset = embeded_unalbeld_dataset,
        batch_size = 256,
    )

    y_pred = []
    y_true = []
    for emb, label in loader_unlabel_embedded:
        emb = emb.to(device)
        out = classfier_network(emb)
        out = out.to("cpu").detach().numpy()
        label = label.to("cpu").detach().numpy()
        y_pred.extend(out)
        y_true.extend(label)

    roc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    pr = average_precision_score(y_true, y_pred)

    print(f'ROC-AUC: {roc}')
    print(f'ROC_PR: {pr}')

    if print_embeddeds:
        gelabeld = np.array(embeded_unlabeld_data)
        lables = np.array(embeded_unlabeld_label)

        tnse = TSNE(n_components=2, random_state=42)
        features_2d = tnse.fit_transform(gelabeld)

        plt.figure(figsize=(10,7))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=lables, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualisierung der 32-dimensionalen Vektoren')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

    if print_learning:
        plt.plot(list_epoch_loss)
        plt.title('Loss Curve Classifier')
        plt.show()
   
    return roc, pr

def bagging_ensamble_training(file: str, 
                  random_seed: int, 
                  percentage_labeld:float = 0.05, 
                  contrastiv_margin: float = 5.0, 
                  lr_siamese: float = 0.0001, 
                  lr_classifier: float = 0.001, 
                  epochs_siamese: int = 20, 
                  epochs_classifier: int = 40, 
                  print_embeddeds: bool = False, 
                  print_learning: int =False,
                  len_dataset: int = 200000,
                  verbose: int = 0,
                  number_learner: int = 5,
                  ):
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    #device: str = "cpu"
    device: torch.device = torch.device(device)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    current_directory: str = os.getcwd() # Current directory ../Masterarbeit//Siames
    parent_dirctory: str = os.path.dirname(current_directory)
    file_classic_dataset: str = "/Archiv/Classical/" # Folder with alle classic ADBench datasets
    file = file

    print(f'##############Start Training with Dataset {file}######################')

    data: np.lib.npyio.NpzFile = np.load(current_directory + file_classic_dataset + file, allow_pickle = True)
    X: np.ndarray = data['X']
    y: np.ndarray = data['y']

    samples, input_features = X.shape

    random_dataset: Random_Dataset = Random_Dataset(x=X, y=y, percent_labeld=percentage_labeld, len_dataset=len_dataset)
    unlabeld_dataset: Simple_Dataset = random_dataset.get_unlabeld_dataset()
    labeld_dataset: Simple_Dataset = random_dataset.get_labeld_dataset()

    loader_random_dataset = DataLoader(
        dataset = random_dataset,
        batch_size = 256,
    )

    loader_unlabeld_data = DataLoader(
        dataset = unlabeld_dataset,
        batch_size = 256,
    )

    loader_label_data = DataLoader(
        dataset = labeld_dataset,
        batch_size = 256,
    )
    if verbose >= 1:
        print(f'Die länge des ungelabendeten Datensatzen ist: {len(unlabeld_dataset)}')
        print(f'Die länge des ungelabendeten Datenloader ist: {len(loader_unlabeld_data)}')

    list_siamese_learner = []
    list_classfier_learner = []
    list_optimizer_siames = []
    list_optimizer_classifier = []
    for _ in range(number_learner):
        # Initialize 5 Models (Siames and Classifier) with their optimizer
        net: Siamese_Network = Siamese_Network(input_size=input_features).to(device)
        classifier: Classifier_Network = Classifier_Network().to(device)
        optimizer_siamese = torch.optim.Adam(net.parameters(), lr=lr_siamese)
        optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=lr_classifier)
        # append all models and optimizer to a list
        list_siamese_learner.append(net)
        list_classfier_learner.append(classifier)
        list_optimizer_siames.append(optimizer_siamese)
        list_optimizer_classifier.append(optimizer_classifier)

    contrastiv_loss: ContrastiveLoss = ContrastiveLoss(margin = contrastiv_margin)
    mse_loss =  F.binary_cross_entropy


    counter_siamese = 0
    for siamese, optimizer in zip(list_siamese_learner, list_optimizer_siames):
        counter_siamese += 1
        print(f'----------------Start trainign Epoche {counter_siamese} Siamese ----------------')
        list_epoch_loss = [] 
        loss_iteration = []
        for epoch in range(epochs_siamese):
            print(f'----------------Start trainign Epoche {epoch}----------------')
            epoch_loss: float = 0.0
            for (inp1, inp2, lable) in loader_random_dataset:
                inp1, inp2, lable = inp1.to(device), inp2.to(device), lable.to(device)
                output1, output2 = siamese(inp1, inp2)
                loss = contrastiv_loss(output1, output2, lable)
                loss.backward()
                optimizer.step()
                iteration_loss = loss.item()
                loss_iteration.append(iteration_loss)
                epoch_loss += loss.item()

            list_epoch_loss.append(epoch_loss/len(loader_random_dataset))
            print(f'Epoche: {epoch} Average Loss: {epoch_loss/len(loader_random_dataset)}')
        print(f"---------Finished Training {counter_siamese} Siamese Network-------------")
        if print_learning:
            plt.plot(list_epoch_loss)
            plt.title('Loss Curve Siamese')
            plt.show()

    counter_classifier = 0
    for siamese, classifier, optimizer in zip(list_siamese_learner, list_classfier_learner, list_optimizer_classifier):
        embeded_labeld_data = []
        embeded_labled_label = []
        for inp, label in loader_label_data:
            inp = inp.to(device)
            out, _ = siamese(inp, inp)
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
        list_epoch_loss = [] 
        loss_iteration = []

        counter_classifier += 1
        for epoch in range(epochs_classifier):
            print(f'----------------Start trainign Epoche {epoch}----------------')
            epoch_loss: float = 0.0
            for embed, lab in loader_label_embedded:
                lab = lab.unsqueeze(1)
                embed, lab = embed.to(device), lab.to(device)

                out = classifier(embed)

                loss = mse_loss(out, lab)
                loss.backward()
                optimizer.step()
                iteration_loss = loss.item()
                loss_iteration.append(iteration_loss)

                epoch_loss += iteration_loss

            list_epoch_loss.append(epoch_loss/len(loader_label_embedded))
            print(f'Epoche: {epoch} Average Loss: {epoch_loss/len(loader_label_embedded)}')
    
        print(f"------------------Finished Training Classifier: {counter_classifier}----------------------")

   
    y_pred = []
    y_true = []
    for inp, lable in loader_unlabeld_data:
        inp = inp.to(device)
        y_outs = []
        for siamese, classifier in zip(list_siamese_learner, list_classfier_learner):
            out_siamese, _ = siamese(inp, inp)
            out = classifier(out_siamese)
            out = out.to("cpu").detach().numpy()
            y_outs.append(out)

        y_out = analyze_lists(y_outs)
        assert(len(y_out) == len(lable))
        y_pred.extend(y_out)
        y_true.extend(lable)

    roc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    pr = average_precision_score(y_true, y_pred)

    print(f'ROC-AUC: {roc}')
    print(f'ROC_PR: {pr}')
    return roc, pr

def analyze_lists(list_of_lists):
    # Ermitteln der maximalen Länge der Listen
    max_length = max(len(lst) for lst in list_of_lists)
    
    # Ergebnisliste initialisieren
    result = []
    
    # Iteriere über die Positionen
    for i in range(max_length):
        count_over = 0
        count_under = 0
        
        # Überprüfen der Werte an der aktuellen Position in jeder Liste
        for lst in list_of_lists:
            if i < len(lst):  # Sicherstellen, dass die Liste lang genug ist
                if lst[i] > 0.5:
                    count_over += 1
                elif lst[i] < 0.5:
                    count_under += 1
        
        # Ergebnis abhängig von den Zählungen
        if count_over >= 3:
            result.append(1)
        elif count_under >= 3:
            result.append(0)
        else:
            result.append(np.nan)  # Keine klare Mehrheit
    
    return result




def stacking_ensamble_training():
    None
  

#if __name__ == "__main__":
#    main()