import torch
import torch.nn as nn

from ..attack import Attack
class SINIFGSM(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://github.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=5, decay=1.0, m=5)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, m=5):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_image = adv_images.detach() + self.decay*self.alpha*momentum
            nes_image.requires_grad = True
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_image = nes_image / torch.pow(2, i)
                outputs = self.model(nes_image)
                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                adv_grad += torch.autograd.grad(cost, nes_image,
                                                retain_graph=False, create_graph=False)[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images