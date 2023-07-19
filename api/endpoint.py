import mlflow


if __name__ == '__main__':

    # To use a remote storage location, provide the HTTP URI, for example:mlflow.set_tracking_uri('<https://my-tracking-server:5000>')
    # To use a local folder to store the logs, prefix the full path with 'file:/' & use it
    mlflow.set_tracking_uri('file:/runs')