from dataclasses import dataclass

import jax

from jax import numpy as jnp

from jax.tree_util import register_pytree_node_class
from jax.experimental.host_callback import id_tap

import wandb

from einops import rearrange


@register_pytree_node_class
@dataclass
class Infos:
    loss_infos: any
    plain_infos: any
    masked_infos: any

    @classmethod
    def init(cls, loss_infos={}, plain_infos={}, masked_infos={}):
        return cls(
            loss_infos=loss_infos,
            plain_infos=plain_infos,
            masked_infos=masked_infos,
        )

    @classmethod
    def merge(cls, a, b):
        return cls.init(
            loss_infos={**a.loss_infos, **b.loss_infos},
            plain_infos={**a.plain_infos, **b.plain_infos},
            masked_infos={**a.masked_infos, **b.masked_infos},
        )

    def add_loss_info(self, name, value):
        return Infos.init(
            loss_infos={**self.loss_infos, name: value},
            plain_infos=self.plain_infos,
            masked_infos=self.masked_infos,
        )

    def add_plain_info(self, name, value):
        return Infos.init(
            loss_infos=self.loss_infos,
            plain_infos={**self.plain_infos, name: value},
            masked_infos=self.masked_infos,
        )

    def add_masked_info(self, name, value, mask):
        return Infos.init(
            loss_infos=self.loss_infos,
            plain_infos=self.plain_infos,
            masked_infos={**self.masked_infos, name: (value, mask)},
        )

    def tree_flatten(self):
        loss_info_names = list(self.loss_infos.keys())
        loss_info_values = list(self.loss_infos.values())

        plain_info_names = list(self.plain_infos.keys())
        plain_info_values = list(self.plain_infos.values())

        masked_info_names = list(self.masked_infos.keys())
        masked_info_values = list(self.masked_infos.values())

        return (
            loss_info_values,
            plain_info_values,
            masked_info_values,
        ), (
            loss_info_names,
            plain_info_names,
            masked_info_names,
        )

    @classmethod
    def tree_unflatten(cls, aux, data):
        loss_info_names, plain_info_names, masked_info_names = aux
        loss_info_values, plain_info_values, masked_info_values = data

        loss_infos = {
            name: value for name, value in zip(loss_info_names, loss_info_values)
        }
        plain_infos = {
            name: value for name, value in zip(plain_info_names, plain_info_values)
        }
        masked_infos = {
            name: value for name, value in zip(masked_info_names, masked_info_values)
        }
        return cls.init(
            loss_infos=loss_infos,
            plain_infos=plain_infos,
            masked_infos=masked_infos,
        )

    def host_get_dict(self):
        def remove_nan(tree):
            return jax.tree_map(lambda x: x[jnp.isnan(x) == False], tree)
        result_dict = {
            **{name: value[mask] for name, (value, mask) in self.masked_infos.items()},
            **remove_nan(self.plain_infos),
            **remove_nan(self.loss_infos),
        }

        return result_dict

    def condense(self, axis=0, method='mean'):
        condenser = None
        if method=='mean':
            condenser = lambda x: jnp.mean(x, axis=axis)
        if method=='unstack':
            condenser = lambda x: rearrange(x, "a b ... -> (a b) ...")
        return Infos.init(
            loss_infos=jax.tree_map(
                condenser,
                self.loss_infos,
            ),
            plain_infos=jax.tree_map(
                condenser,
                self.plain_infos,
            ),
            masked_infos=jax.tree_map(
                condenser,
                self.masked_infos,
            ),
        )

    def host_dump_to_wandb(self):
        wandb.log(self.host_get_dict())

    def dump_to_wandb(self):
        id_tap(lambda arg, _: Infos.host_dump_to_wandb(arg), self)

    def host_get_str(self):
        loss_msg = "Losses:" + "".join(
            [f"\n\t{name}: {value}" for name, value in self.loss_infos.items()]
        )
        info_msg = "Infos:" + "".join(
            [f"\n\t{name}: {value}" for name, value in self.plain_infos.items()]
        )

        return loss_msg + "\n" + info_msg

    def host_dump_to_console(self):
        return # print(self.host_get_str())

    def dump_to_console(self):
        id_tap(lambda arg, _: Infos.host_dump_to_console(arg), self)
