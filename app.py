from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve
from fast_bert.prediction import BertClassificationPredictor
from google.cloud import storage
import traceback
from utils import label_map, process_text
from typing import List
import numpy as np
from time import time
import os
import sys


def create_app(model):
    app = Flask(__name__)
    CORS(app)

    @app.route('/', methods=['GET'])
    def ping():
        print('pong')
        return jsonify({'message': 'Activity Classifier app is running'}), 200

    @app.route('/', methods=['POST'])
    def clusteringtask():
        print('Classification endpoint hit')
        start = time()

        try:
            data = request.json['text']
            if type(data) == str:
                data = [data]

            data = process_text(data)
            pred = model.predict_batch(data)

            return jsonify({
                'message': 'Classification successful',
                'classification': [label_map(int(x[0][0])) for x in pred],
                'enum': [int(x[0][0]) for x in pred],
                'confidence': [x[0][1] for x in pred],
                'time': time() - start
            }), 200
        except Exception as e:
            tb = traceback.format_exc()
            print(f"TRACEBACK:\n\n{tb}\n")
            return jsonify({'message': str(e), 'stacktrace': str(tb)}), 500

    return app


if __name__ == '__main__':
    path = 'models/model_out/pytorch_model.bin'
    bucket_path = 'https://storage.cloud.google.com/boast-trained-models/activity_classifier/pytorch_model.bin'

    # fetch model from google storage if not exist
    if bucket_path is not None and not os.path.exists(path):
        # set env key
        if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_auth.json'

        client = storage.Client()
        bucket = client.get_bucket('boast-trained-models')
        blob = bucket.get_blob('activity_classifier/pytorch_model.bin')

        print('Downloading model...')
        with open(path, 'wb') as file_obj:
            blob.download_to_file(file_obj)

    predictor = BertClassificationPredictor(
        model_path='models/model_out',
        label_path='train',
        multi_label=False,
        model_type='distilbert',
        do_lower_case=True)

    serve(create_app(predictor), host='0.0.0.0', port=5000)
