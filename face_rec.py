import numpy as np
import pandas as pd
import cv2
import redis
import os
import time
from datetime import datetime


# insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# connect to redis client
hostname = 'redis-19067.c61.us-east-1-3.ec2.cloud.redislabs.com'
portnumber = 19067
password = 'FTJyVLQp6mCvEJt08TNDm7YrnJvH5hu3'

r = redis.StrictRedis(host=hostname, port=portnumber, password=password)


def retreive_data(name):

    retreive_dict = r.hgetall(name)
    retreive_series = pd.Series(retreive_dict)
    retreive_series = retreive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32
                                                                    ))
    index = retreive_series.index
    index = list(map(lambda x: x.decode(), index))

    retreive_series.index = index
    retreive_df = retreive_series.to_frame().reset_index()
    retreive_df.columns = ['name_role', 'facial_features']
    retreive_df[['Name', 'Role']] = retreive_df['name_role'].apply(lambda x:
                                                                   x.split('@')).apply(pd.Series)
    return retreive_df[['Name', 'Role', 'facial_features']]


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc', root='insight_model', providers=[
                       'CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# ml algorithm function


def ml_search_algorithm(dataframe, feature_column, name_role, test_vector, thresh=0.5):

    dataframe = dataframe.copy()
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    data_filter = dataframe.query(f'cosine>={thresh}')
    if (len(data_filter) > 0):
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
    return person_name, person_role

# multiple face recognition


class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def saveLogs_redis(self):
        dataframe = pd.DataFrame(self.logs)

        dataframe.drop_duplicates('name', inplace=True)
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()

        encoded_data = []

        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush("attendance-logs", *encoded_data)

        self.reset_dict()

    def face_recognition(self, test_image, dataframe, feature_column, name_role, thresh=0.5):

        current_time = str(datetime.now())

        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(
                dataframe, feature_column, name_role=name_role, test_vector=embeddings, thresh=thresh)

            if person_name == 'Unknown':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)

            text_gen = person_name
            cv2.putText(test_copy, text_gen, (x1, y1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_copy, current_time, (x1, y2+10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy


class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embeddings(self, frame):

        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            text = f"samples={self.sample}"
            cv2.putText(frame, text, (x1, y1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)
            embeddings = res['embedding']

        return frame, embeddings

    def save_data_in_redis_db(self, name, role):
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        # step1 - load face_embedding.txt
        # step2 - convert into array
        # step3 - cal. mean embeddings
        # step4 - save this into redis database

        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)

        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)

        x_array = np.asarray(x_array)

        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)

        x_mean_bytes = x_mean.tobytes()

        r.hset(name='academy:register', key=key, value=x_mean_bytes)

        os.remove('face_embedding.txt')

        self.reset()

        return True
