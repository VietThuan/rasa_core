from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import six

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter, RasaNLUHttpInterpreter

def run_babi(serve_forever=True):
    agent = Agent.load("examples_weather/babi/models/policy/weather",
                       interpreter=
                           RasaNLUHttpInterpreter(model_name='model_20171013-084449', token=None,
                                                  server='http://localhost:7000'))

    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    run_babi()
