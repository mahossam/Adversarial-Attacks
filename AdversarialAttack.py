from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from sympy.abc import epsilon

EPSILONS = [0.1, 0.15, 0.2, 0.25, 0.3]

class AdversarialAttack(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def attack(self, data, target_class):
        pass

class FGSMTargeted(AdversarialAttack):
    def __init__(self, model, device, epsilons=EPSILONS):
        super(FGSMTargeted, self).__init__(model, device)
        self.epsilons = epsilons
        self._loss = nn.CrossEntropyLoss()

    def _perturb(self, data, epsilon, gradient):
        data_perturbed = data - epsilon * gradient.sign()

        # keep image data in the [0,1] range
        data_perturbed = torch.clamp(data_perturbed, 0, 1)
        return data_perturbed

    def attack(self, data, target_class):
        attack_is_successful = False
        attack_epsilon = None

        data = data.to(self.device)

        for epsilon in self.epsilons:

            data.requires_grad = True

            output = self.model(data)
            init_pred = output.argmax(dim=1, keepdim=True)

            loss = self._loss(output, torch.tensor([target_class], dtype=torch.long))

            self.model.zero_grad()
            loss.backward()
            data_gradient = data.grad.data

            perturbed_data = self._perturb(data, epsilon, data_gradient)

            # predict class for adversarial sample
            adv_output = self.model(perturbed_data)
            adv_pred = adv_output.argmax(dim=1, keepdim=True)

            if adv_pred.item() == target_class:
                attack_is_successful = True
                attack_epsilon = epsilon
                break

        return perturbed_data, attack_is_successful, adv_pred, attack_epsilon
