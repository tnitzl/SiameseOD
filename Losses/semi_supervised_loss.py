import torch


class Semi_Supervised_Loss(torch.nn.Module):
    def __init__(self):
        super(Semi_Supervised_Loss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, student, teacher, targets, weight):
        unsupervised_loss = 0.0
        supervised_loss = 0.0

        count_unsupervies = 0
        count_supervies = 0
        for x1, x2, tar in zip(student, teacher, targets):
            if tar == -1:
                # print(x1)
                # print(x2)
                loss = weight * torch.nn.functional.mse_loss(x1, x2)
                # print(loss)
                unsupervised_loss += loss
                count_unsupervies += 1
            else:
                x1 = torch.nn.functional.sigmoid(x1)
                loss = torch.nn.functional.binary_cross_entropy(x1, tar)
                # print(loss)
                supervised_loss += loss
                count_supervies += 1

        # print(f'Die unsupervies loss ist: {unsupervised_loss/count_unsupervies}')
        # print(f'Die supervies loss ist: {supervised_loss/count_supervies}')
        if count_supervies > 0 and count_unsupervies > 0:
            return (
                (
                    unsupervised_loss / count_unsupervies
                    + supervised_loss / count_supervies
                )
                * 0.5,
                unsupervised_loss / count_unsupervies,
                supervised_loss / count_supervies,
            )
        elif count_unsupervies > 0:
            return (
                unsupervised_loss / count_unsupervies,
                unsupervised_loss / count_unsupervies,
                torch.tensor(0.0),
            )
        elif count_supervies > 0:
            return (
                supervised_loss / count_supervies,
                torch.tensor(0.0),
                supervised_loss / count_supervies,
            )
        else:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
