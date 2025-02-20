"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Type

from pydantic import BaseModel, PrivateAttr, field_validator

from .base_node import BaseNode


class MCTSNode(BaseNode):

    prior: float = 0.0
    c_puct: float = 1.5

    __visit_count: int = PrivateAttr(default=0)
    __value_sum: float = PrivateAttr(default=0)

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0
        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update_visit_count(self, count: int) -> None:
        self.__visit_count = count

    def get_all_prior(self) -> float:
        all_prior = 0
        node = self
        while node:
            all_prior += node.prior
            node = node.parent
        return all_prior

    def update(self, value: float) -> None:
        # init value
        # if self.value == -100:
        #     self.value = value
        self.__visit_count += 1
        # self.__value_sum += value
        # if self.parent:
        #     kl_penalty_reward = self.prior - self.parent.get_all_Prio
        # else:
        #     kl_penalty_reward = 0
        # self.__value_sum += value + kl_penalty_reward * 0.1 # BETA
        # self.__value_sum += value - self.get_all_prior()
        self.__value_sum += value


    def update_recursive(self, value: float, start_node: Type[BaseNode]) -> None:
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)


    def puct(self) -> float:
        q_value = self.q_value() if self.visit_count() > 0 else 0
        if self.value == -100:
            action_prior = self.prior - self.parent.prior
            # u_value = self.c_puct * np.exp(self.prior / self.state['length']) * np.sqrt(self.parent.visit_count()) / (1 + self.visit_count())
            u_value = self.c_puct * np.exp(action_prior / self.state['length']) * np.sqrt(self.parent.visit_count()) / (1 + self.visit_count())
        else:
            u_value = self.c_puct * self.value * np.sqrt(self.parent.visit_count()) / (1 + self.visit_count())
        return q_value + u_value
