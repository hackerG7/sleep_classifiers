import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_collection import HeartRateCollection
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService


class HeartRateFeatureService(object):
    WINDOW_SIZE = 10 * 30 - 15

    @staticmethod
    def load(subject_id):
        heart_rate_feature_path = HeartRateFeatureService.get_path(subject_id)
        feature = pd.read_csv(str(heart_rate_feature_path), delimiter=' ').values
        return feature

    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_hr_feature.out')

    @staticmethod
    def write(subject_id, feature):
        heart_rate_feature_path = HeartRateFeatureService.get_path(subject_id)
        np.savetxt(heart_rate_feature_path, feature, fmt='%f')

    @staticmethod
    def build(subject_id, valid_epochs):
        heart_rate_collection = HeartRateService.load_cropped(subject_id)
        return HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)

    @staticmethod
    def build_from_collection(heart_rate_collection, valid_epochs):
        heart_rate_features = []
        interpolated_timestamps, interpolated_hr = HeartRateFeatureService.interpolate_and_normalize(
            heart_rate_collection)

        for epoch in valid_epochs:
            #print("interpolated_timestamps.shape", interpolated_timestamps.shape)
            #print("interpolated_timestamps: ", interpolated_timestamps)

            #print("epoch: ", epoch)
            indices_in_range = HeartRateFeatureService.get_window(interpolated_timestamps, epoch)
            heart_rate_values_in_range = interpolated_hr[indices_in_range]
            #print("indices_in_range: ", indices_in_range)
            #print("heart_rate_values_in_range.shape ", heart_rate_values_in_range.shape)
            #print("heart_rate_values_in_range: ", heart_rate_values_in_range)
            feature = HeartRateFeatureService.get_feature(heart_rate_values_in_range)
            print("feature: ", feature)

            heart_rate_features.append(feature)

        return np.array(heart_rate_features)

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - HeartRateFeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION + HeartRateFeatureService.WINDOW_SIZE
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def get_feature(heart_rate_values):
        #print("heart_rate_values: ", heart_rate_values)
        #print("standardized: ", np.std(heart_rate_values))
        return [np.std(heart_rate_values)]

    @staticmethod
    def interpolate_and_normalize(heart_rate_collection):
        print("heart_rate_collection.timestamps.shape ", heart_rate_collection.timestamps.shape)
        print("heart_rate_collection.timestamps: ", heart_rate_collection.timestamps)
        print("heart_rate_collection.timestamps.flatten(): ", heart_rate_collection.timestamps.flatten())
        print("heart_rate_collection.values.shape: ", heart_rate_collection.values.shape)
        print("heart_rate_collection.values: ", heart_rate_collection.values)
        print("heart_rate_collection.values.flatten(): ", heart_rate_collection.values.flatten())
        timestamps = heart_rate_collection.timestamps.flatten()
        heart_rate_values = heart_rate_collection.values.flatten()
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_hr = np.interp(interpolated_timestamps, timestamps, heart_rate_values)
        print("interpolated_hr.shape: ", interpolated_hr.shape)
        print("interpolated_hr: ", interpolated_hr)
        print("interpolated_hr, np.interp: ", interpolated_hr)
        interpolated_hr = utils.convolve_with_dog(interpolated_hr, 10)#HeartRateFeatureService.WINDOW_SIZE)
        print("interpolated_hr, convolve_with_dog.shape: ", interpolated_hr.shape)
        print("interpolated_hr, convolve_with_dog: ", interpolated_hr)
        #print("interpolated_hr, convolve_with_dog: ", interpolated_hr)

        scalar = np.percentile(np.abs(interpolated_hr), 90)
        print("scalar: ", scalar)
        interpolated_hr = interpolated_hr / scalar
        print("interpolated_hr, scalar: ", interpolated_hr)

        return interpolated_timestamps, interpolated_hr
HeartRateFeatureService.interpolate_and_normalize(HeartRateCollection("123123",np.array([ 
    [0, 87],
    [3, 82],
    [5, 80],
    [9, 70],
    [11, 68],
    [13, 67],
    [15, 66],
    [17, 65],
    [19, 65],
    [20, 65],
    [22, 65],
    [24, 65],
    [27, 65],
    [30, 65]
])
    ))