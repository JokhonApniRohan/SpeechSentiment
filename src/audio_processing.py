import opensmile
import pandas as pd

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_egemaps(path):
    try:
        features = smile.process_file(path)
        return features
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return pd.DataFrame()
