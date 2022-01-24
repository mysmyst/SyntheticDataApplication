from gretel_client import create_project
from gretel_client.helpers import poll
import yaml
from smart_open import open
import json
from getpass import getpass
import pandas as pd
from gretel_client import configure_session, ClientConfig

pd.set_option('max_colwidth', None)

configure_session(ClientConfig(api_key="grtu2ada0d43cbd6812cae24777aa6f925074a1b2c861bf2b3489fe9ee7901337b8b",
                               endpoint="https://api.gretel.cloud"))


project = create_project(display_name="synthetic-data")


with open("input.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set the model epochs to 50
config['models'][0]['synthetics']['params']['epochs'] = 25

print(json.dumps(config, indent=2))

# Load and preview the DataFrame to train the synthetic model on.

dataset_path = 'https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv'
df = pd.read_csv(dataset_path)
df.to_csv('training_data.csv', index=False)
df


model = project.create_model_obj(model_config=config)
model.data_source = 'training_data.csv'
model.submit(upload_data_source=True)

poll(model)


synthetic_df = pd.read_csv(model.get_artifact_link(
    "data_preview"), compression='gzip')

synthetic_df.to_csv("synthetic_data.csv")
