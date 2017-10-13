from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from examples_weather.weather_policy import WeatherPolicy
from rasa_core.agent import Agent
from rasa_core.channels.file import FileInputChannel
from rasa_core.interpreter import RegexInterpreter, RasaNLUHttpInterpreter
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)


def run_babi_online():
    training_data = 'data/weather.md'
    logger.info("Starting to train policy")
    agent = Agent("../weather_domain.yml",
                  policies=[MemoizationPolicy(), WeatherPolicy()],
                  interpreter=RegexInterpreter())

    input_c = FileInputChannel(training_data,
                               message_line_pattern='^\s*\*\s(.*)$',
                               max_messages=10)
    agent.train_online(training_data,
                       input_channel=input_c,
                       epochs=10)

    agent.interpreter = RasaNLUHttpInterpreter(model_name='model_20171013-084449', token=None,
                                                  server='http://localhost:7000')
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_babi_online()
