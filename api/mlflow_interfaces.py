import os
import sys

import mlflow
from mlflow.tracking import MlflowClient



HOME = 'customer-complaints-email-classification-case-study'
curr = os.path.dirname(__file__)
while True:
    if curr.endswith(HOME):
        break
    curr = os.path.dirname(curr)
sys.path.append(curr)

from utils import get_credential



class MlflowTrackingServer:

    def __init__(self, host=""):
        self.host = host
        mlflow.set_tracking_uri(f"http://{self.host}:5000")

    def createif_experiment(self, name):
        mlflow.set_experiment(name)



if __name__ == '__main__':

    host_file = os.path.join(curr, 'doc', 'mlflow.tracking_server.host.txt')

    tracking_server = MlflowTrackingServer(host=get_credential(host_file))

    tracking_server.createif_experiment("experiment-002")

