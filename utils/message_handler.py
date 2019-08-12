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
        credentials = pika.PlainCredentials('guest', 'guest')
        parameters =  pika.ConnectionParameters(config.RABBITMQ_CONNECTION, credentials=credentials, heartbeat=5)
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.threads = []
        
    
    def _publish(self, queue, body):
        # Should use separate connection to publish and comsume
        publisher = MessageHandler(config.RABBITMQ_CONNECTION)
        publisher.channel.basic_publish(
                exchange='', routing_key=queue, body=body)
        print(f" [x] Sent {body} to queue: {queue}")
        print("Close publisher")
        publisher.close()
    
    def sendMessage(self, queue, body):
        try:
            self._publish(queue, body)
        except Exception as e:
            print("Stop with Error:", e)

    def consumeMessage(self, queue, callback):
        on_message_callback = partial(self.on_message, args=(self.connection, self.threads, callback))
        self.channel.basic_consume(queue=queue,on_message_callback=on_message_callback)
        print(f' [*] Waiting for messages from {queue}. To exit press CTRL+C')
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
        
        # Wait for all to complete
        for thread in self.threads:
            thread.join()
        
        self.connection.close()

    def ack_message(self, channel, delivery_tag):
        if channel.is_open:
            channel.basic_ack(delivery_tag)

    def on_message(self, channel, method_frame, header_frame, body, args):
        (connection, threads, callback) = args
        delivery_tag = method_frame.delivery_tag
        t = threading.Thread(target=self.do_work, args=(connection, channel, delivery_tag, body, method_frame, header_frame, callback))
        t.start()
        threads.append(t)

    def do_work(self, connection, channel, delivery_tag, body,  method_frame, header_frame, callback):
        thread_id = threading.get_ident()
        fmt1 = 'Thread id: {} Delivery tag: {} Message body: {}'
        
        # Run callback function to train model, and ...
        callback(channel, method_frame, header_frame, body)
        
        cb = partial(self.ack_message, channel, delivery_tag)
        connection.add_callback_threadsafe(cb)

    def close(self):
        self.connection.close()

Message_Handler = MessageHandler(config.RABBITMQ_CONNECTION)