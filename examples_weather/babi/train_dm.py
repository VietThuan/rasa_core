from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from examples_weather.Custom_Policy import CustomMemoizationPolicy
from examples_weather.weather_policy import WeatherPolicy
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy


def train_babi_dm():
    training_data_file = 'data/weather.md'
    model_path = 'models/policy/weather'

    agent = Agent("../weather_domain.yml",
                  policies=[MemoizationPolicy(),WeatherPolicy()])

    agent.train(
            training_data_file,
            max_history=1,
            epochs=2000,
            batch_size=50,
            augmentation_factor=50,
            validation_split=0.2
    )

    agent.persist(model_path)


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    train_babi_dm()
