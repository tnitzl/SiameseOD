import torch

# https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
class ContrastiveLoss(torch.nn.Module):
    # weighted loss
    def __init__(self, margin=20.0, verbose=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = 1.0
        # for debugging
        self.verbose = verbose

    def forward(self, student, teacher, label):

        # euclidean_distance = torch.sqrt((output1 - output2) ** 2)

        distance = torch.nn.functional.pairwise_distance(student, teacher)

        if self.verbose >= 1:
            print(
                f"distance between the two vektores {student}-{teacher}: {distance}"
            )
        # negativ_differenz = self.margin ** 2 - euclidean_distance ** 2
        # if self.verbose >= 1:
        #    print(negativ_differenz)

        bound_negativ_pair = (
            torch.max(self.margin**2 - distance, torch.tensor(0.0)) ** 2
        )
        if self.verbose >= 1:
            print(f"bound negative pair:{bound_negativ_pair}")

        # https://dvl.in.tum.de/slides/adl4cv-ws20/2.Siamese.pdf
        loss_contrastive = (
            label * distance**2 + (1 - label) * bound_negativ_pair
        )
        if self.verbose >= 1:
            print(f"contrastive loss: {loss_contrastive}")
        return loss_contrastive.mean()
    

class ContrastiveLoss_cosinesimularity(torch.nn.Module):
    # weighted loss
    def __init__(self, margin=20.0, verbose=0):
        super(ContrastiveLoss_cosinesimularity, self).__init__()
        self.margin = margin
        # for debugging
        self.verbose = verbose

    def forward(self, student, teacher, label):

        # euclidean_distance = torch.sqrt((output1 - output2) ** 2)

        distance = torch.nn.functional.cosine_similarity(student, teacher, dim=1)
        new_distance = 1 - distance

        if self.verbose >= 1:
            print(
                f"distance between the two vektores {student}-{teacher}: {new_distance}"
            )
        # negativ_differenz = self.margin ** 2 - euclidean_distance ** 2
        # if self.verbose >= 1:
        #    print(negativ_differenz)

        bound_negativ_pair = (
            torch.max(self.margin**2 - new_distance, torch.tensor(0.0)) ** 2
        )
        if self.verbose >= 1:
            print(f"bound negative pair:{bound_negativ_pair}")

        # https://dvl.in.tum.de/slides/adl4cv-ws20/2.Siamese.pdf
        loss_contrastive = (
            label * new_distance**2 + (1 - label) * bound_negativ_pair
        )
        if self.verbose >= 1:
            print(f"contrastive loss: {loss_contrastive}")
        return loss_contrastive.mean()