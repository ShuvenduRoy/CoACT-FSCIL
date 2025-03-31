"""Encoder model for FSCIL."""

import argparse
from typing import Any, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from models.model_utils import get_model_args


def print_trainable_parameters(model: Any) -> None:
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}",
    )


class EncoderWrapper(nn.Module):
    """Encoder Wrapper encoders."""

    def __init__(
        self,
        args: argparse.Namespace,
        pet_config: Optional[Any] = None,
    ) -> None:
        """Initialize the EncoderWrapper.

        Parameters
        ----------
        args : argparse.ArgumentParser
            Arguments passed to the model.
        pet_config : Any
            PET configuration.

        Returns
        -------
        None
        """
        super(EncoderWrapper, self).__init__()
        print("Loading encoder: ", args.hf_model_checkpoint)

        self.args = args
        model_args = get_model_args(args.hf_model_checkpoint)
        self.num_features = model_args["embedding_dim"]

        print(f"Loading model from {args.hf_model_checkpoint}")
        self.model = model_args["hf_model_class"].from_pretrained(
            args.hf_model_checkpoint,
        )

        # Freeze pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        layers = []
        for name, _ in self.model.named_parameters():
            layers.append(name)
        print(f"Layers in the model {args.hf_model_checkpoint}: ", len(layers))

        if pet_config is not None:
            print("Adding PET moduels.")
            self.model = get_peft_model(self.model, pet_config)

            layers = []
            for name, _ in self.model.named_parameters():
                layers.append(name)
            print(
                f"Layers in the model with PETF modules of {args.hf_model_checkpoint}: ",
                len(layers),
            )

        if args.num_mlp == 1:
            self.mlp = nn.Sequential(nn.Linear(self.num_features, self.args.moco_dim))
        elif args.num_mlp == 2:
            self.mlp = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
                nn.Linear(self.num_features, self.args.moco_dim),
            )
        else:
            self.mlp = nn.Sequential(nn.Identity())

        self.classifier = nn.Linear(
            self.num_features,
            self.args.num_classes,
            bias=self.args.add_bias_in_classifier,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            patch embeddings, [b, embed_dim]
            projecting embedding, [b, moco_dim]
            output logits, [b, n_classes]
        """
        output = self.model(x)
        output = output.pooler_output  # [b, embed_dim]
        return (
            output,  # [b, embed_dim=768]
            self.mlp(output),  # [b, moco_dim=128] or [b, embed_dim] if no mlp layer
            self.classifier(output),  # [b, n_classes]
        )


class FSCILencoder(nn.Module):
    """FSCIL Model class."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize FSCIL Model.

        Parameters
        ----------
        args : argparse.ArgumentParser
            Arguments passed to the model.

        Returns
        -------
        None
        """
        super().__init__()

        self.args = args
        config = self._get_pet_config()
        self.encoder_q = EncoderWrapper(args, pet_config=config)
        self.num_features = self.encoder_q.num_features

        print_trainable_parameters(self.encoder_q)
        self.encoder_k = EncoderWrapper(args)

        # Make the starting parameters of student and teacher identical
        # Handle the case that EMA can be different in terms of parameters
        encoder_q_params = self._get_encoder_q_state_dict()
        for name, param in self.encoder_k.named_parameters():
            param.requires_grad = False
            if name in encoder_q_params:
                param.data.copy_(encoder_q_params[name])
            else:
                print(f"Warning: {name} not found in encoder_q")

        # print param name and status
        print("\nencoder_q parameters:")
        for name, param in self.encoder_q.named_parameters():
            print("model.encoder_q", name, param.requires_grad)

        # group the parameters into pre-trained and newly added parameter
        # with different learning rates for optimization
        pet_name = "none" if self.args.pet_cls is None else self.args.pet_cls.lower()
        self.params_with_lr = [
            {
                "params": [
                    p
                    for n, p in self.encoder_q.named_parameters()
                    if pet_name in n or n.startswith("classifier")
                ],
                "lr": args.lr_base,
            },
            {
                "params": [
                    p
                    for n, p in self.encoder_q.named_parameters()
                    if pet_name not in n and not n.startswith("classifier")
                ],
                "lr": args.lr_base * args.encoder_lr_factor,
            },
        ]

        # print the parameters with learning rate
        lr_dict = {
            args.lr_base: [
                n
                for n, p in self.encoder_q.named_parameters()
                if pet_name in n or n.startswith("classifier")
            ],
        }
        encoder_params = [
            n
            for n, p in self.encoder_q.named_parameters()
            if pet_name not in n and not n.startswith("classifier")
        ]
        if args.encoder_lr_factor == 1:
            lr_dict[args.lr_base].extend(encoder_params)
        else:
            lr_dict[args.lr_base * args.encoder_lr_factor] = encoder_params

        layers_in_lr_dict = []
        for key, val in lr_dict.items():
            layers_in_lr_dict.extend(val)
            for v in val:
                print(v, key)

        parameter_names = [n for n, p in self.encoder_q.named_parameters()]
        assert len(layers_in_lr_dict) == len(parameter_names), "Layers mismatch"
        for n in parameter_names:
            if n not in layers_in_lr_dict:
                print(n, args.lr_base)

    def _get_pet_config(self) -> Optional[Any]:
        """Get the PET configuration.

        Returns
        -------
        Optional[Any]
            The PET configuration.
        """
        if self.args.pet_cls is None:
            return None
        if self.args.pet_cls.lower() == "lora":
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                layers_to_transform=self.args.adapt_blocks,
                bias="none",
            )
        return config

    def _get_encoder_q_state_dict(self) -> dict:
        """Get the state dictionary of the encoder_q.

        PEFT added additional wrapper, making the parameter
        names different from the original model.
        The original names are needed in few places,
        e.g. when updating the teacher encoder.

        Returns
        -------
        dict
            The state dictionary of the encoder_q.
        """
        encoder_q_params_original = self.encoder_q.state_dict()
        encoder_q_params = {}
        # remove prefix 'base_model.model.' and "base_layer." from keys
        for key, value in encoder_q_params_original.items():
            if "base_model.model." in key:
                key = key.replace("base_model.model.", "")  # noqa: PLW2901
            if "base_layer." in key:
                key = key.replace("base_layer.", "")  # noqa: PLW2901
            encoder_q_params[key] = value
        return encoder_q_params

    @torch.no_grad()
    def _momentum_update_key_encoder(self, base_sess: bool) -> None:
        """Momentum update of the key encoder.

        Parameters
        ----------
        base_sess : bool
            Whether the current session is a base session.

        Returns
        -------
        None
        """
        encoder_q_params = self._get_encoder_q_state_dict()
        for name, param in self.encoder_k.named_parameters():
            param.data = param.data * self.args.moco_m + encoder_q_params[name] * (
                1.0 - self.args.moco_m
            )

    def forward(
        self,
        im_q: torch.Tensor,
        im_k: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        base_sess: bool = True,
    ) -> Any:
        """Forward pass of the model.

        Parameters
        ----------
        im_q : torch.Tensor, optional
            The query image, or Input image at test time
        im_k : torch.Tensor, optional
            The key image, by default None.
        labels : torch.Tensor, optional
            The labels, by default None.
        base_sess : bool, optional
            Whether the current session is a base session, by default True.

        Returns
        -------
        Any
            The output tensor.
        """
        if isinstance(im_q, list):
            im_q = torch.cat(im_q, dim=0)
        token_embeding, embedding_q, logits = self.encoder_q(
            im_q,
        )  # [b, embed_dim=768], [b, moco_dim=128], [b, n_classes]

        if labels is None:  # during evaluation, im_q should be a single image
            return (token_embeding, logits)
        embedding_q = nn.functional.normalize(embedding_q, dim=1)

        # foward key
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder(base_sess)  # update the key encoder
            _, embedding_k, _ = self.encoder_k(im_k)  # keys: bs x dim
            embedding_k = nn.functional.normalize(embedding_k, dim=1)

        if embedding_q.shape[0] != embedding_k.shape[0]:  # multiple views
            embedding_q = embedding_q.view(
                embedding_k.shape[0],
                -1,
                embedding_k.shape[1],
            )
        else:
            embedding_q = embedding_q.unsqueeze(1)
        embedding_k = embedding_k.unsqueeze(1)

        return logits[: embedding_q.shape[0]], embedding_q, embedding_k
