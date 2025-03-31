"""FSCIL training module."""

import argparse
from copy import deepcopy
from typing import Tuple

import torch

from dataloaders.helpter import get_dataloader
from losses.contrastive import SupConLoss
from methods.helper import (
    base_train_one_epoch,
    get_optimizer_base,
    inc_train_one_epoch,
    replace_fc_with_prototypes,
    test,
)
from models.encoder import FSCILencoder, print_trainable_parameters
from utils.train_utils import ensure_path


class FSCITTrainer:
    """FSCIL Trainer class."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize FSCIL Trainer.

        Parameters
        ----------
        args : argparse.ArgumentParser
            Arguments passed to the trainer.

        Returns
        -------
        None
        """
        self.args = args
        ensure_path(args.save_path)

        # initialize model
        self.model = FSCILencoder(args)
        self.pre_trained_model = deepcopy(self.model)
        self.criterion = SupConLoss()
        self.optimizer, self.scheduler = get_optimizer_base(self.model, self.args)
        self.device_id = None
        self.best_model_dict = deepcopy(self.model.state_dict())

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            self.pre_trained_model = self.pre_trained_model.cuda()

    def adjust_learnable_parameters(  # noqa: PLR0912
        self,
        session: int,
        epoch: int,
    ) -> None:
        """Adjust the learnable parameters base of config and current step.

        Parameters
        ----------
        session: int
            Current session
        epoch: int
            Current epoch
        """
        # handle trainable parameters in the base session
        # at certain epochs
        if session == 0:
            if (
                epoch == self.args.encoder_ft_start_epoch
            ):  # current epoch is ft start epoch
                print_trainable_parameters(self.model.encoder_q)

                status = self.args.encoder_ft_start_layer == -1  # full fine-tune
                for (
                    name,
                    param,
                ) in self.model.encoder_q.named_parameters():
                    if "layer." in name:
                        # find number in name with regex
                        layer_num = int("".join(filter(str.isdigit, name)))
                        if layer_num == self.args.encoder_ft_start_layer:
                            status = True  # start fine-tuning from this layer

                    # update the requires_grad status is not already trainable
                    param.requires_grad = status or param.requires_grad

                # print the status of the encoder
                for (
                    name,
                    param,
                ) in self.model.encoder_q.named_parameters():
                    print(
                        "ecnoder_q @session {} @epoch {},".format(session, epoch),
                        name,
                        param.requires_grad,
                    )

                print_trainable_parameters(self.model.encoder_q)

        # handle trainable parameters for incremental sessions
        elif self.args.incft:
            if "pet" in self.args.incft_layers:
                for name, param in self.model.encoder_q.named_parameters():
                    if self.args.pet_cls.lower() in name:
                        param.requires_grad = True
            if "classifier" in self.args.incft_layers:
                for param in self.model.encoder_q.classifier.parameters():
                    param.requires_grad = True
            for name, param in self.model.encoder_q.named_parameters():
                print(
                    "ecnoder_q @session {} @epoch {},".format(session, epoch),
                    name,
                    param.requires_grad,
                )
            print_trainable_parameters(self.model.encoder_q)

    def update_matrix(self, accuracies: Tuple, session: int) -> None:
        """Update the accuracy matrix.

        Parameters
        ----------
        accuracies: Tuple
            Tuple of accuracies for base, incremental and all classes.
        session: int
            Current session
        """
        base_acc, inc_acc, all_acc = accuracies

        self.session_accuracies["base"][session] = max(
            base_acc,
            self.session_accuracies["base"][session],
        )
        self.session_accuracies["incremental"][session] = max(
            inc_acc,
            self.session_accuracies["incremental"][session],
        )
        self.session_accuracies["all"][session] = max(
            all_acc,
            self.session_accuracies["all"][session],
        )

    def train(self) -> None:
        """Train the model."""
        self.session_accuracies = {
            "base": [0] * self.args.sessions,
            "incremental": [0] * self.args.sessions,
            "all": [0] * self.args.sessions,
        }
        for session in range(self.args.sessions):
            # initialize dataset
            prototype_loader, trainloader, testloader = get_dataloader(
                self.args,
                session,
            )

            # train session
            print(f"Training session {session}...")
            print(f"Train set size: {len(trainloader.dataset)}")
            print(f"Test set size: {len(testloader.dataset)}")

            if session == 0:  # base session
                if self.args.start_training_with_prototypes:
                    # replace base classifier weight with prototypes
                    print("Replacing base classifier weight with prototypes...")
                    replace_fc_with_prototypes(
                        prototype_loader,
                        self.model,
                        self.args,
                        self.device_id,
                    )
                for epoch in range(self.args.epochs_base):
                    # adjust learnable params
                    self.adjust_learnable_parameters(session, epoch)

                    # train and test
                    base_train_one_epoch(
                        model=self.model,
                        trainloader=trainloader,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        args=self.args,
                        device_id=self.device_id,
                    )
                    self.scheduler.step()
                    base_acc, inc_acc, all_acc = test(
                        model=self.model,
                        testloader=testloader,
                        epoch=epoch,
                        args=self.args,
                        session=session,
                        device_id=self.device_id,
                    )

                    if all_acc > self.session_accuracies["all"][session]:
                        self.best_model_dict = deepcopy(
                            self.model.state_dict(),
                        )

                    self.update_matrix((base_acc, inc_acc, all_acc), session)

                # --- END OF BASE SESSION ---
                # load the best saved model for the base session
                self.model.load_state_dict(self.best_model_dict)

                if self.args.update_base_classifier_with_prototypes:
                    # replace base classifier weight with prototypes
                    print("Replacing base classifier weight with prototypes...")
                    replace_fc_with_prototypes(
                        prototype_loader,
                        self.model,
                        self.args,
                        self.device_id,
                    )
                    base_acc, inc_acc, all_acc = test(
                        model=self.model,
                        testloader=testloader,
                        epoch=self.args.epochs_base,
                        args=self.args,
                        session=session,
                        device_id=self.device_id,
                    )

                    self.update_matrix((base_acc, inc_acc, all_acc), session)

                # make a copy of the base session for dual consistency learning
                self.base_model = deepcopy(self.model)

                # By default, turn off the learnable parameters of the model
                for param in self.model.parameters():
                    param.requires_grad = False

                # reset the optimizer with lr = lr * args.inc_ft_lr_factor
                self.model.params_with_lr = [
                    {
                        "params": [
                            p for n, p in self.model.encoder_q.named_parameters()
                        ],
                        "lr": self.args.lr_base * self.args.inc_ft_lr_factor,
                    },
                ]
                self.optimizer, _ = get_optimizer_base(
                    self.model,
                    self.args,
                )

            else:
                self.adjust_learnable_parameters(session, 0)
                print("Replacing inc. classifier weight with prototypes...")
                replace_fc_with_prototypes(
                    prototype_loader,
                    self.model,
                    self.args,
                    self.device_id,
                )
                for epoch in range(self.args.epochs_incremental):
                    inc_train_one_epoch(
                        model=self.model,
                        model_base=self.base_model,
                        model_pretrained=self.pre_trained_model,
                        trainloader=trainloader,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        args=self.args,
                        device_id=self.device_id,
                    )
                base_acc, inc_acc, all_acc = test(
                    model=self.model,
                    testloader=testloader,
                    epoch=0,
                    args=self.args,
                    session=session,
                    device_id=self.device_id,
                )
                self.update_matrix((base_acc, inc_acc, all_acc), session)
            print(f"Session {session} completed.")
            print("Base acc: ", self.session_accuracies["base"])
            print("Inc. acc: ", self.session_accuracies["incremental"])
            print("Overall : ", self.session_accuracies["all"])
