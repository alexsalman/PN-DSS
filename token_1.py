#!/usr/bin/env python
import requests

def get_patient_data(patient_id):
    user_data = {
        'token': '08FA6317C61592BFED9DC8C0D62028A1',
        'content': 'record',
        'records[0]': patient_id,
        'format': 'json',
        'returnFormat': 'json'
    }
    r = requests.post('https://redcap.idissc.org/redcap/api/', data=user_data)
    return r.json()

if __name__ == "__main__":
    patient_id = '20189'  # Replace with a real record ID from REDCap
    response = get_patient_data(patient_id)
    print(response)