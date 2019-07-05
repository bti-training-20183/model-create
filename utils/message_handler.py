import sys
import os
sys.path.append(os.getcwd())
import config
import pika


class MessageHandler:
    def __init__(self, host):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(config.RABBITMQ_CONNECTION))
        self.channel = self.connection.channel()
        self.channel.queue_declare('from_client')
        self.channel.queue_declare('from_creator')
        self.channel.queue_declare('from_preprocessor')
        self.channel.queue_declare('from_deployer')

    def sendMessage(self, queue, body):
        self.channel.basic_publish(
            exchange='', routing_key=queue, body=body)
        print(f" [x] Sent {body} to queue: {queue}")

    def consumeMessage(self, queue, callback):
        self.channel.basic_consume(
            queue=queue, on_message_callback=callback, auto_ack=True)
        print(f' [*] Waiting for messages from {queue}. To exit press CTRL+C')
        self.channel.start_consuming()


MessageHdlr = MessageHandler(config.RABBITMQ_CONNECTION)
