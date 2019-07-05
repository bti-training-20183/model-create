from os import environ

RABBITMQ_CONNECTION = environ["RABBITMQ_CONNCETION"] if environ.get(
    "RABBITMQ_CONNECTION") else "localhost"
QUEUE = {
    "from_client" : "from_client",
    "from_preprocessor" : "from_preprocessor",
    "from_creator" : "from_creator",
    "from_deployer" : "from_deployer"
}

