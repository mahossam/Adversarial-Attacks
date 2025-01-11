from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

EPSILONS = [0.1, 0.15, 0.2, 0.25, 0.3]

class AdversarialAttack(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def generate_adv_example(self, data, target_class):
        """ An abstract method to generate an adversarial example.

        Implement this method in a subclass to perform a specific adversarial attack method.
        """
        pass

class FGSMTargeted(AdversarialAttack):
    def __init__(self, model, device, epsilons=EPSILONS):
        super(FGSMTargeted, self).__init__(model, device)
        self.epsilons = epsilons
        self._loss = nn.CrossEntropyLoss()

    def _perturb(self, data, epsilon, gradient) -> torch.Tensor:
        """Apply FGSM adversarial perturbation to the input data."""
        data_perturbed = data - epsilon * gradient.sign()

        # keep image data in the [0,1] range
        data_perturbed = torch.clamp(data_perturbed, 0, 1)
        return data_perturbed

    def generate_adv_example(self, data, target_class) -> Tuple[torch.Tensor, bool, torch.Tensor, float]:
        """Generate an adversarial example using the input data to a target class.

        Parameters
        ----------
        data : torch.Tensor
            The input data to be used to generate the adversarial attack.
        target_class : int
            The target class for the adversarial attack.

        Returns
        -------
        perturbed_data : torch.Tensor
            The adversarial example if successful, otherwise the original data.
        attack_is_successful : bool
            Whether the attack was successful.
        adv_pred : torch.Tensor
            The predicted class of the perturbed data.
        attack_epsilon : float
            The epsilon value used for the successful attack.
        """
        attack_is_successful = False
        perturbed_data = data
        attack_epsilon = None
        adv_pred = None

        data = data.to(self.device)

        for epsilon in self.epsilons:
            data.requires_grad = True

            output = self.model(data)

            loss = self._loss(output, torch.tensor(data=[target_class], dtype=torch.long))

            self.model.zero_grad()
            loss.backward()
            data_gradient = data.grad.data

            perturbed_data = self._perturb(data=data, epsilon=epsilon, gradient=data_gradient)

            # predict class for the perturbed data
            adv_output = self.model(perturbed_data)
            adv_pred = adv_output.argmax(dim=1, keepdim=True)

            if adv_pred.item() == target_class:
                attack_is_successful = True
                attack_epsilon = epsilon
                break

        return perturbed_data, attack_is_successful, adv_pred, attack_epsilon
