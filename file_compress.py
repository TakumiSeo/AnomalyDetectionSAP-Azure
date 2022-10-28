# Import related packages:
import os
from datetime import datetime
from azure.storage.blob import BlobClient, BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import zipfile

BLOB_SAS_TEMPLATE = "https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"


def zip_file(root, name):
    z = zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED)
    for f in os.listdir(root):
        if (f.endswith("csv")) & ("train" not in f):
            z.write(os.path.join(root, f), f)
    z.close()

def upload_to_blob(file, conn_str, container, blob_name):
    blob_client = BlobClient.from_connection_string(conn_str, container_name=container, blob_name=blob_name)
    with open(file, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

def generate_data_source_sas(conn_str, container, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    sas_token = generate_blob_sas(account_name=blob_service_client.account_name,
                                  container_name=container,
                                  blob_name=blob_name,
                                  account_key=blob_service_client.credential.account_key,
                                  permission=BlobSasPermissions(read=True),
                                  expiry=datetime.utcnow() + timedelta(days=2))
    blob_sas = BLOB_SAS_TEMPLATE.format(account_name=blob_service_client.account_name,
                                        container_name=container,
                                        blob_name=blob_name,
                                        sas_token=sas_token)
    return blob_sas