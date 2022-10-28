import json, os
from datetime import datetime, timedelta
import base64
from flask import Blueprint, Response, request
from flask_cors import cross_origin
import pandas as pd
import json
import subprocess
from dotenv import load_dotenv
load_dotenv()
import requests
import numpy as np
import librosa
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.ai.anomalydetector.models import ModelInfo
from azure.core.credentials import AzureKeyCredential
from file_compress import zip_file, upload_to_blob, generate_data_source_sas
from firebase import delete_firebase_doc
import shutil
import os
from io import BytesIO

# Generate Router Instance
router = Blueprint('router', __name__)
@router.route("/post_ac_values", methods=["POST"])
@cross_origin(supports_credentials=True)
def post_ac_values():
  '''
  post acclerometer values as vibValues
  return danger level 
  '''
  param = json.loads(request.data.decode('utf-8'))
  df = make_input_df(param.get("vibValues"))
  received_model_id = param.get("model_id")
  payload = synchronised_request(df, param.get("calibration"))
  header = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": "{subscription_key}".format(subscription_key=os.getenv("anomaly-detector-subscription-key"))}
  API_MODEL_LAST_INFERENCE = "{endpoint}/multivariate/models/{model_id}/last/detect"
  url = API_MODEL_LAST_INFERENCE.format(endpoint= os.getenv("anomaly-detector-endpoint") + "/anomalydetector/v1.1-preview.1", model_id=received_model_id)
  res = requests.post(url, headers=header, data=json.dumps(payload))
  res_content = json.loads(res.content.decode('utf-8'))
  res_list = []
  for item in res_content.get("results"):
      score = item.get("value")
      res_list.append({"is_anomaly": score.get("isAnomaly"),
                      "severity": score.get("severity"),
                      "score": score.get("score")})
  # 4 categories 25% 50 % 75% <=100%
  df_res = pd.DataFrame(res_list)
  danger_level = severity_out(df_res)
  return Response(response=json.dumps({"danger_level": danger_level}), status=200)

@router.route("/post_audio_values", methods=["POST"])
@cross_origin(supports_credentials=True)
def post_audio_values():
  '''
  post audio values as vibValues
  return danger level 
  '''
  param = json.loads(request.data.decode('utf-8'))
  content_string = param.get("base64Audio")
  time_value = param.get("timestamp")
  # model id that is pretrained in Azure Anomaly Detection
  received_model_id = param.get("model_id")
  # for the test in notebook
  with open("audio_anom.txt", mode='w') as f:
    f.write(content_string)
  f.close()
  # zero crossing
  # df = zero_crossing(content_string, time_value, training=False)
  # # make body of zero crossing
  # payload = audio_payload(df)
  df = effect_hpss(content_string, time_value, window=100, slide_window=50, training=False)
  payload = audio_payload_effect(df)
  header = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": "{subscription_key}".format(subscription_key=os.getenv("anomaly-detector-subscription-key"))}
  API_MODEL_LAST_INFERENCE = "{endpoint}/multivariate/models/{model_id}/last/detect"
  url = API_MODEL_LAST_INFERENCE.format(endpoint= os.getenv("anomaly-detector-endpoint") + "/anomalydetector/v1.1-preview.1", model_id=received_model_id)
  res = requests.post(url, headers=header, data=json.dumps(payload))
  res_content = json.loads(res.content.decode('utf-8'))
  print(res_content)
  res_list = []
  for item in res_content.get("results"):
      score = item.get("value")
      res_list.append({"is_anomaly": score.get("isAnomaly"),
                      "severity": score.get("severity"),
                      "score": score.get("score")})

  df_res = pd.DataFrame(res_list)
  # break down danger level into 4 categories 25% 50 % 75% 100%
  danger_level = severity_out(df_res)
  return Response(response=json.dumps({"danger_level": danger_level}), status=200)

@router.route("/post_audio", methods=["POST"])
@cross_origin(supports_credentials=True)
def post_audio_training_data():
  '''
  train audio data
  return model id
  '''
  param = json.loads(request.data.decode('utf-8'))
  content_string = param.get("base64Audio")
  # Write the gained data down txt file 
  # with open("audio_anom.txt", mode='w') as f:
  #   f.write(content_string)
  # f.close()
  time_value = param.get("timestamp")
  # Data Preprocessing phases
  # 1: Zero crossing 
  # df = zero_crossing(content_string, time_value, training=True)
  # 2: percussion and harmony
  df = effect_hpss(content_string, time_value, window=100, slide_window=50, training=True)
  res = train_audio_model(df)
  return Response(response=json.dumps({"model_ID": res.get("model_id"), "status": res.get("status")}), status=200)


@router.route("/post_trainig_data", methods=["POST"])
@cross_origin(supports_credentials=True)
def post_trainig_data():
  '''
  train meter data
  return model id
  '''
  param = json.loads(request.data.decode('utf-8'))
  posted_data = param.get("reading")
  calib_values = param.get("calib")
  df = pd.DataFrame.from_dict(posted_data)
  df.x = df["x"].apply(lambda x: x - calib_values.get("x"))
  df.y = df["y"].apply(lambda y: y - calib_values.get("y"))
  df.z = df["z"].apply(lambda z: z  - calib_values.get("z"))
  min_time = datetime.strptime(df.timestamp[0][:19]+"Z",  '%Y-%m-%dT%H:%M:%SZ')
  for i in range(len(df)):
      df.timestamp[i] = min_time + timedelta(seconds=i)
  df.timestamp = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ"))
  df.timestamp = df["timestamp"].apply(lambda x: x.replace(" ", "T"))
  df = df.set_index('timestamp', drop=True)
  res = train_model(df)
  return Response(response=json.dumps({"model_ID": res.get("model_id"), "status": res.get("status")}), status=200)


def train_model(df):
  subscription_key = os.getenv("anomaly-detector-subscription-key")
  anomaly_detector_endpoint = os.getenv("anomaly-detector-endpoint")
  connection_key = os.getenv("azure-connection-key")
  source_folder = "train_data"
  zipfile_name = "train"
  account_name = "anomditectstorage"   # storage account name
  resource_group = "sap_anomdict"  # resource group
  try:
    cmd = f"az storage account keys list -g {resource_group} -n {account_name}"   # using az-cli is safer
    az_response = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE).stdout
    key = json.loads(az_response)[0]["value"]
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={key};EndpointSuffix=core.windows.net"
  except FileNotFoundError:    # no az-cli available
      connection_string = os.getenv("STORAGE_CONN_STR")

  os.makedirs(source_folder, exist_ok=True)
  for variable in df.columns:
      individual_df = pd.DataFrame(df[variable].values, index=df.index, columns=["value"])
      individual_df.to_csv(os.path.join(source_folder, f"{variable}.csv"))

  # connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
  connection_string = "DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net".format(account_name, connection_key)
  container_name = "sap-anom-container"
  zipfile_name = "train.zip"
  zip_file(source_folder, zipfile_name)
  upload_to_blob(zipfile_name, connection_string, container_name, zipfile_name)
  data_source = generate_data_source_sas(connection_string, container_name, zipfile_name)
  print(data_source)
  ad_client = AnomalyDetectorClient(AzureKeyCredential(subscription_key), anomaly_detector_endpoint)
  start_time = df.index[0]#"2022-10-07T12:45:29.34Z"
  end_time = df.index[-1]#"2022-10-07T12:49:00.64Z"
  sliding_window = 500
  data_feed = ModelInfo(start_time=start_time, end_time=end_time, source=data_source, sliding_window=sliding_window)
  response_header = ad_client.train_multivariate_model(data_feed, cls=lambda *args: [args[i] for i in range(len(args))])[-1]
  trained_model_id = response_header['Location'].split("/")[-1]
  # print(f"model id: {trained_model_id}")
  model_status = ad_client.get_multivariate_model(trained_model_id).model_info.status
  # print(f"model status: {model_status}")
  return {"model_id": trained_model_id, "status": model_status}

def train_audio_model(df):
  subscription_key = os.getenv("anomaly-detector-subscription-key")
  anomaly_detector_endpoint = os.getenv("anomaly-detector-endpoint")
  connection_key = os.getenv("azure-connection-key")
  source_folder = "train_audio_data"
  zipfile_name = "train_audio"
  account_name = "anomditectstorage"
  resource_group = "sap_anomdict"
  try:
    cmd = f"az storage account keys list -g {resource_group} -n {account_name}"
    az_response = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE).stdout
    key = json.loads(az_response)[0]["value"]
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={key};EndpointSuffix=core.windows.net"
  except FileNotFoundError:
      connection_string = os.getenv("STORAGE_CONN_STR")

  os.makedirs(source_folder, exist_ok=True)
  for variable in df.columns:
      individual_df = pd.DataFrame(df[variable].values, index=df.index, columns=["value"])
      individual_df.to_csv(os.path.join(source_folder, f"{variable}.csv"))
  # works on local
  # connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
  connection_string = "DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net".format(account_name, connection_key)
  container_name = "sap-anom-container"
  zipfile_name = "train_audio.zip"
  zip_file(source_folder, zipfile_name)
  upload_to_blob(zipfile_name, connection_string, container_name, zipfile_name)
  data_source = generate_data_source_sas(connection_string, container_name, zipfile_name)
  print(data_source)
  ad_client = AnomalyDetectorClient(AzureKeyCredential(subscription_key), anomaly_detector_endpoint)
  start_time = df.index[0]
  end_time = df.index[-1]
  sliding_window = 500
  data_feed = ModelInfo(start_time=start_time, end_time=end_time, source=data_source, sliding_window=sliding_window)
  response_header = ad_client.train_multivariate_model(data_feed, cls=lambda *args: [args[i] for i in range(len(args))])[-1]
  trained_model_id = response_header['Location'].split("/")[-1]
  # print(f"model id: {trained_model_id}")
  model_status = ad_client.get_multivariate_model(trained_model_id).model_info.status
  # print(f"model status: {model_status}")
  return {"model_id": trained_model_id, "status": model_status}

@router.route("/model_status", methods=["POST"])
@cross_origin(supports_credentials=True)
def get_model_status():
  '''
  see if a pretrained model is available or not
  return model status
  '''
  param = json.loads(request.data.decode('utf-8'))
  model_id = param.get("model_id")
  print(model_id)
  subscription_key = os.getenv("anomaly-detector-subscription-key")
  anomaly_detector_endpoint = os.getenv("anomaly-detector-endpoint")
  ad_client = AnomalyDetectorClient(AzureKeyCredential(subscription_key), anomaly_detector_endpoint)
  model_status = ad_client.get_multivariate_model(model_id).model_info.status
  print(model_status)
  if (model_status == "RUNNING") or (model_status=="CREATED"):
    model_status = "Trainig, wait for few minutes"
  elif (model_status=="READY"):
    model_status = "Available this model"
  return Response(response=json.dumps({"status": model_status}), status=200)

@router.route("/delete_trained_data", methods=["DELETE"])
@cross_origin(supports_credentials=True)
def delete_trained_data():
  param = json.loads(request.data.decode('utf-8'))
  model_id = param.get("model_id")
  model_id = str(model_id)
  ENDPOINT = os.getenv("anomaly-detector-endpoint") + "anomalydetector/v1.1-preview.1"
  HEADERS = {
      "Ocp-Apim-Subscription-Key": os.getenv("anomaly-detector-subscription-key")
  }
  API_DELETE = "{endpoint}/multivariate/models/{model_id}"
  res = requests.delete(API_DELETE.format(endpoint=ENDPOINT, model_id=model_id), headers=HEADERS)
  # delete db data
  delete_firebase_doc(model_id)
  assert res.status_code == 204, f"Error occured. Error message: {res.content}"
  return Response(response=json.dumps({"message": "deleted"}), status=200)

@router.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', '*')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

def make_input_df(docs):
  df_init = pd.DataFrame.from_dict(docs)
  df = df_init.copy()
  df = df.sort_values('timestamp')
  df.timestamp = df["timestamp"].apply(lambda x: x[:-2]+"Z")
  df.timestamp = df["timestamp"].apply(lambda x: x.replace(" ", "T"))
  df = df.set_index('timestamp', drop=True)
  return df

def zero_crossing(content_string, time_value, training=False):
  decoded = base64.b64decode(content_string)
  with open('dummy.' + "wav", 'wb') as file:
      shutil.copyfileobj(BytesIO(decoded), file, length=131072)
  soundwave_sf, _ = librosa.load('dummy.' +"wav", sr=22050)
  os.remove('dummy.' + "wav")
  time_value = datetime.strptime(time_value[:19]+"Z",  '%Y-%m-%dT%H:%M:%SZ')
  time_list = [time_value]
  # Zero Crossing Rate
  soundwave_sf = librosa.feature.zero_crossing_rate(y=soundwave_sf, frame_length=128, hop_length=32, center=True)
  soundwave_sf = soundwave_sf[0]
  for i in range(len(soundwave_sf) - 1):
      time_list.append(time_list[0] + timedelta(seconds=i))
  df = pd.DataFrame([time_list, soundwave_sf]).T
  df.columns = ["timestamp", "value"]
  if training:
    df = df.set_index('timestamp', drop=True)
  else:
    pass
  return df

def effect_hpss(content_string, time_value, window, slide_window, training=False):
  decoded = base64.b64decode(content_string)
  with open('dummy.' + "wav", 'wb') as file:
      shutil.copyfileobj(BytesIO(decoded), file, length=131072)
  soundwave_sf, _ = librosa.load('dummy.' +"wav", sr=22050)
  os.remove('dummy.' + "wav")
  
  # make a time series data
  time_value = datetime.strptime(time_value[:19]+"Z",  '%Y-%m-%dT%H:%M:%SZ')
  time_list = [time_value]
  
  # devide audio into percussion and harmony
  y_harm, y_perc = librosa.effects.hpss(soundwave_sf)
  harm_mean = []
  harm_var = []
  per_mean = []
  per_var = []
  last_window = len(y_harm)
  for sl in range(0, len(y_harm), slide_window):
      if (last_window-sl) - window < 0:
          harm_mean.append(np.mean(y_harm[sl:-1]))
          harm_var.append(np.std(y_harm[sl:-1]))
          per_mean.append(np.mean(y_perc[sl:-1]))
          per_var.append(np.std(y_perc[sl:-1]))
      else:
          harm_mean.append(np.mean(y_harm[sl:sl + window]))
          harm_var.append(np.std(y_harm[sl:sl + window]))
          per_mean.append(np.mean(y_perc[sl:sl + window]))
          per_var.append(np.std(y_perc[sl:sl + window]))
  for i in range(len(harm_mean) - 1):
      time_list.append(time_list[0] + timedelta(seconds=i))
  df = pd.DataFrame([time_list, harm_mean, harm_var, per_mean, per_var]).T
  df.columns = ["timestamp", "hm_val", "hv_val", "pm_val", "pv_val"]
  
  if training:
    df = df.set_index('timestamp', drop=True)
  else:
    pass

  return df

def synchronised_request(df, calib):
  # print(df)
  timestamp_list = df.index.tolist()
  x_calib = calib.get("x")
  y_calib = calib.get("y")
  z_calib = calib.get("z")
  x_val = [x - x_calib for x in df.x.tolist()]
  y_val = [y - y_calib for y in df.y.tolist()]
  z_val = [z - z_calib for z in df.z.tolist()]
  return {
      "variables": [
          {
              "name": "x",
              "timestamps": timestamp_list,
              "values": x_val
          },
                {
              "name": "y",
              "timestamps": timestamp_list,
              "values": y_val
          },
                {
              "name": "z",
              "timestamps": timestamp_list,
              "values": z_val
          }
      ],
      "detectingPoints": 10
  }

def audio_payload_effect(df):
  df.timestamp = df.timestamp.apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ"))
  df.timestamp = df.timestamp.apply(lambda x: x.replace(" ", "T"))
  timestamp_list = df.timestamp.tolist()
  # df.value = df.value.apply(lambda x: np.round(float(x), 10))
  df = df.fillna(0)
  res =  {
      "variables": [
          {
              "name": "hm_val",
              "timestamps": timestamp_list,
              "values": df.hm_val.tolist()
          },
             {
              "name": "hv_val",
              "timestamps": timestamp_list,
              "values": df.hv_val.tolist()
          },
          {
              "name": "pm_val",
              "timestamps": timestamp_list,
              "values": df.pm_val.tolist()
          },
             {
              "name": "pv_val",
              "timestamps": timestamp_list,
              "values": df.pv_val.tolist()
          },
          
          
      ],
      "detectingPoints": 10
  }
  return res

def audio_payload(df):
  df.timestamp = df.timestamp.apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ"))
  df.timestamp = df.timestamp.apply(lambda x: x.replace(" ", "T"))
  timestamp_list = df.timestamp.tolist()
  df.value = df.value.apply(lambda x: np.round(float(x), 10))
  df = df.fillna(0)
  val = df.value.tolist()
  res =  {
      "variables": [
          {
              "name": "value",
              "timestamps": timestamp_list,
              "values": val
          }
      ],
      "detectingPoints": 10
  }
  return res

def severity_out(df):
  '''
  0~0.3   => Green
  0.3~0.5 => Yellow
  0.5~0.7 => Orange
  0.7~1.0 => Red
  '''
  severity_mean = np.mean(df.severity.tolist())
  if severity_mean < 0.3:
    res = "Green"
  elif 0.3 <= severity_mean < 0.6:
    res = "Yellow"
  elif 0.6 <= severity_mean < 0.8:
    res = "Orange"
  elif 0.8 <= severity_mean:
    res = "Red"
  else:
    res = ""
  return res
   
