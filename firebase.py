import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import timedelta

def delete_firebase_doc(model_id):
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except:
        pass
    db = firestore.client()
    docs = db.collection(u'userinfo').where('model_id', '==', model_id).stream()
    for doc in docs:
        doc_id = doc.id
        db.collection(u'userinfo').document(doc_id).delete()