import torch

import torch.nn as nn
import torch.nn.functional as F

class Siamese_Network(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(Siamese_Network, self).__init__()

        #old settings
        #self.fc1 = nn.Linear(input_size, 20)
        #self.fc2 = nn.Linear(20, 32)
        #new settings
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)

    def forward_once(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def forward(self, x, y) -> (torch.Tensor, torch.Tensor):
        out1: torch.Tensor = self.forward_once(x)
        out2: torch.Tensor = self.forward_once(y)
        return out1, out2
    

class Classifier_Network(nn.Module):
    def __init__(self) -> None:
        super(Classifier_Network, self).__init__()
        ## old seetings
        #self.fc1 = nn.Linear(32, 10)
        #self.fc2 = nn.Linear(10, 1)
        ## new seetings
        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
class Stacked_Classifier(nn.Module):
    def __init__(self, number_previous_worker) -> None:
        super(Stacked_Classifier, self).__init__()
        self.fc1 = nn.Linear(number_previous_worker * 32, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


#https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246

class ContrastiveLoss(nn.Module):
 
    def __init__(self, margin=20.0, verbose=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        #for debugging
        self.verbose = verbose

    def forward(self, output1, output2, label):

        #euclidean_distance = torch.sqrt((output1 - output2) ** 2)
        
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        if self.verbose >= 1:
            print(f'distance between the two vektores {output1}-{output2}: {euclidean_distance}')
        #negativ_differenz = self.margin ** 2 - euclidean_distance ** 2
        #if self.verbose >= 1:
        #    print(negativ_differenz)


        bound_negativ_pair = torch.max(self.margin ** 2 - euclidean_distance, torch.tensor(0.0)) ** 2
        if self.verbose >= 1:
            print(f'bound negative pair:{bound_negativ_pair}')

       #https://dvl.in.tum.de/slides/adl4cv-ws20/2.Siamese.pdf 
        loss_contrastive = label * euclidean_distance ** 2 + (1 - label) * bound_negativ_pair
        if self.verbose >= 1:
            print(f'contrastive loss: {loss_contrastive}')    
        return loss_contrastive.mean()