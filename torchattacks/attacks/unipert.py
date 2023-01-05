import numpy as np
import random
import logging
import types

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from typing import Any, Dict, Optional, Union, TYPE_CHECKING
from ..utils import check_and_transform_label_format, get_labels_np_array, projection
from ..attacks.deepfool import DeepFool
from ..attack import Attack

logger = logging.getLogger(__name__)

class UniversalPerturbation(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2016). Computes a fixed perturbation to be applied to all
    future inputs. To this end, it can use any adversarial attack method.
    | Paper link: https://arxiv.org/abs/1610.08401
    """

    def __init__(
        self,
        model: torch.nn.Module,
        attacker: Attack,
        delta: float = 0.2,
        max_iter: int = 20,
        eps: float = 10.0,
        norm: Union[int, float, str] = np.inf,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        """
        :param classifier: A trained classifier.
        :param attacker: Adversarial attack from torchattacks.attacks. Default is DeepFool.
        :param delta: desired accuracy
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :param eps: Attack step size (input variation).
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 2.
        :param batch_size: Batch size for model evaluations in UniversalPerturbation.
        :param verbose: Show progress bars.
        """
        super().__init__("UNIPERT", model)
        self.attacker = attacker
        if (self.attacker is None):
            self.attacker = DeepFool(model)
        self.delta = delta
        self.max_iter = max_iter
        self.eps = eps
        self.norm = norm
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

        # Attack properties
        self._fooling_rate: Optional[float] = None
        self._converged: Optional[bool] = None
        self._noise: Optional[np.ndarray] = None

    @property
    def fooling_rate(self) -> Optional[float]:
        """
        The fooling rate of the universal perturbation on the most recent call to `generate`.
        :return: Fooling Rate.
        """
        return self._fooling_rate

    @property
    def converged(self) -> Optional[bool]:
        """
        The convergence of universal perturbation generation.
        :return: `True` if generation of universal perturbation has converged.
        """
        return self._converged

    @property
    def noise(self) -> Optional[np.ndarray]:
        """
        The universal perturbation.
        :return: Universal perturbation.
        """
        return self._noise

    def forward(
        self, images: np.ndarray, labels: Optional[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        logger.info(
            "Computing universal perturbation based on %s attack.", self.attacker
        )
        classes_num = None

        if labels is not None:
            classes_num = torch.max(labels) + 1
            labels = check_and_transform_label_format(labels, classes_num)

        if labels is None:
            # Use model predictions as true labels
            logger.info("Using model predictions as true labels.")
            labels = get_labels_np_array(
                self.model(images, batch_size=self.batch_size)
            )
            classes_num = torch.max(labels) + 1

        if classes_num == 2 and labels.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        y_index = np.argmax(labels, axis=1)

        # Init universal perturbation
        noise = np.zeros_like(images[[0]])
        fooling_rate = 0.0
        nb_instances = len(images)

        # Instantiate the middle attacker
        attacker = self.attacker

        # Generate the adversarial examples
        nb_iter = 0
        pbar = tqdm(
            total=self.max_iter, desc="Universal perturbation", disable=not self.verbose
        )

        while fooling_rate < 1.0 - self.delta and nb_iter < self.max_iter:
            # Go through all the examples randomly
            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for j, ex in enumerate(images[rnd_idx]):
                x_i = ex[None, ...]

                current_label = np.argmax(self.model(x_i + noise)[0])
                original_label = y_index[rnd_idx][j]

                if current_label == original_label:
                    # Compute adversarial perturbation
                    adv_xi = attacker.forward([x_i + noise], [labels[rnd_idx][[j]]])[0]
                    new_label = np.argmax(self.model(adv_xi)[0])

                    # If the class has changed, update v
                    if current_label != new_label:
                        noise = adv_xi - x_i

                        # Project on L_p ball
                        noise = projection(noise, self.eps, self.norm)
            nb_iter += 1
            pbar.update(1)

            # Apply attack and clip
            x_adv = images + noise
            # if self.estimator.clip_values is not None:
            #     clip_min, clip_max = self.estimator.clip_values
            #     x_adv = np.clip(x_adv, clip_min, clip_max)

            # Compute the error rate
            y_adv = np.argmax(self.model(x_adv, batch_size=1), axis=1)
            fooling_rate = np.sum(y_index != y_adv) / nb_instances

        pbar.close()
        self._fooling_rate = fooling_rate
        self._converged = nb_iter < self.max_iter
        self._noise = noise
        logger.info(
            "Success rate of universal perturbation attack: %.2f%%", 100 * fooling_rate
        )

        return x_adv

    def _get_attack(
        self, a_name: str, params: Optional[Dict[str, Any]] = None
    ) -> Attack:
        """
        Get an attack object from its name.
        :param a_name: Attack name.
        :param params: Attack params.
        :return: Attack object.
        :raises NotImplementedError: If the attack is not supported.
        """
        try:
            attack_class = self._get_class(self.attacks_dict[a_name])
            a_instance = attack_class(self.estimator)  # type: ignore

            if params:
                a_instance.set_params(**params)

            return a_instance
        except KeyError:
            raise NotImplementedError(f"{a_name} attack not supported") from KeyError

    @staticmethod
    def _get_class(class_name: str) -> types.ModuleType:
        """
        Get a class module from its name.
        :param class_name: Full name of a class.
        :return: The class `module`.
        """
        sub_mods = class_name.split(".")
        module_ = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
        class_module = getattr(module_, sub_mods[-1])

        return class_module

    def _check_params(self) -> None:
        if not isinstance(self.delta, (float, int)) or self.delta < 0 or self.delta > 1:
            raise ValueError("The desired accuracy must be in the range [0, 1].")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.eps, (float, int)) or self.eps <= 0:
            raise ValueError("The eps coefficient must be a positive float.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The batch_size must be a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
