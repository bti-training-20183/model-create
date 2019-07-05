import os
import sys
import config
sys.path.append(os.getcwd())
from utils.message_handler import MessageHdlr


def callback(channel, method, properties, body):
    print(f'[x] Received {body} from {properties}')
    MessageHdlr.sendMessage('from_creator', 'Dummy message from deployer')


class Creator:
    def __init__(self):
        pass

    def listen(self, queue):
        MessageHdlr.consumeMessage(queue, callback)

if __name__ == "__main__":
    model_creator = Creator()
    model_creator.listen(config.QUEUE["from_preprocessor"])