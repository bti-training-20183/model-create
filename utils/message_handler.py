import pika
import config
import sys
import os
import socket
import time
import threading
from functools import partial
sys.path.append(os.getcwd())


class MessageHandler:
    def __init__(self, host):
        isreachable = False
        while isreachable is False:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect(('rabbitmq', 5672))
                print("Connected to RabbitMQ")
                isreachable = True
            except socket.error as e:
                print("Not connected to RabbitMQ")
                time.sleep(5)
            s.close()
        if isreachable:
            self._connect()
            self.channel.queue_declare('from_client')
            self.channel.queue_declare('from_creator')
            self.channel.queue_declare('from_preprocessor')
            self.channel.queue_declare('from_deployer')
    
    def _connect(self):
        self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(config.RABBITMQ_CONNECTION))
        self.channel = self.connection.channel()
    
    def _publish(self, queue, body):
        self.channel.basic_publish(
                exchange='', routing_key=queue, body=body)
        print(f" [x] Sent {body} to queue: {queue}")
    
    def sendMessage(self, queue, body):
        try:
            self._publish(queue, body)
        except Exception:
            print('\n\nReconnecting to queue\n\n-------------------------------')
            self._connect()
            self._publish(queue, body)
            
    def on_request(self, channel, method, properties, body, callback):
        callback_thread = threading.Thread(target=callback, args=(channel, method, properties, body))
        callback_thread.start()

    def consumeMessage(self, queue, callback):
        self.channel.basic_consume(
            queue=queue, on_message_callback=partial(self.on_request, callback=callback), auto_ack=True)
        print(f' [*] Waiting for messages from {queue}. To exit press CTRL+C')
        self.channel.start_consuming()

    def close(self):
        self.connection.close()

Message_Handler = MessageHandler(config.RABBITMQ_CONNECTION)