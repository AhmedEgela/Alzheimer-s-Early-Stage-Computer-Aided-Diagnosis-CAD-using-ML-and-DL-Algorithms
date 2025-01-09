import pandas as pd
import SimpleITK as sitk
import streamlit as st
import pickle
import os

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.feature_selection import RFECV


def image_feature_extractor(img_path):

# classical features extraction using PyRadiomics packages
  df_img = pd.DataFrame()

  img = sitk.ReadImage(img_path)

  # firstorder features extraction module
  from radiomics.firstorder import RadiomicsFirstOrder
  _1stOrder = RadiomicsFirstOrder(img, img)
  _1stOrder.enableAllFeatures()
  computed_features1 = _1stOrder.execute()
  for key, value in computed_features1.items():
    df_img[f"{key}"] = pd.Series(value)

  # shape2D features extraction module
  from radiomics.shape2D import RadiomicsShape2D
  shape2d = RadiomicsShape2D(img, img)
  shape2d.enableAllFeatures()
  computed_features2 = shape2d.execute()
  for key, value in computed_features2.items():
    df_img[f"{key}"] = pd.Series(value)

  # glcm features extraction module
  from radiomics.glcm import RadiomicsGLCM
  RadiomicsGLCM = RadiomicsGLCM(img, img)
  RadiomicsGLCM.enableAllFeatures()  # Enables all first-order features
  computed_features3 = RadiomicsGLCM.execute()
  for key, value in computed_features3.items():
    df_img[f"{key}"] = pd.Series(value)

  # glrlm features extraction module
  from radiomics.glrlm import RadiomicsGLRLM
  RadiomicsGLRLM = RadiomicsGLRLM(img, img)
  RadiomicsGLRLM.enableAllFeatures()
  computed_features4 = RadiomicsGLRLM.execute()
  for key, value in computed_features4.items():
    df_img[f"{key}"] = pd.Series(value)

  # ngtdm features extraction module
  from radiomics.ngtdm import RadiomicsNGTDM
  RadiomicsNGTDM = RadiomicsNGTDM(img, img)
  RadiomicsNGTDM.enableAllFeatures()
  computed_features5 = RadiomicsNGTDM.execute()
  for key, value in computed_features5.items():
    df_img[f"{key}"] = pd.Series(value)

  # gldm features extraction module
  from radiomics.gldm import RadiomicsGLDM
  RadiomicsGLDM = RadiomicsGLDM(img, img)
  RadiomicsGLDM.enableAllFeatures()
  computed_features6 = RadiomicsGLDM.execute()
  for key, value in computed_features6.items():
    df_img[f"{key}"] = pd.Series(value)

  # glszm features extraction module
  from radiomics.glszm import RadiomicsGLSZM
  RadiomicsGLSZM = RadiomicsGLSZM(img, img)
  RadiomicsGLSZM.enableAllFeatures()
  computed_features7 = RadiomicsGLSZM.execute()
  for key, value in computed_features7.items():
    df_img[f"{key}"] = pd.Series(value)

  return df_img


def main():
    this_dir = os.getcwd()
    
    pickle_in = open(os.path.join(this_dir, "Deployed_App/classifier.pkl"),"rb")
    classifier = pickle.load(pickle_in)

    scaler_in = open(os.path.join(this_dir, "Deployed_App/model_scaler.pkl"),"rb")
    scaler = pickle.load(scaler_in)

    encoder_in = open(os.path.join(this_dir, "Deployed_App/model_label_encoder.pkl"),"rb")
    encoder = pickle.load(encoder_in)
    st.title("Alzheimer Early Diagnosis [MRI Modality]")
    file_uploader = st.file_uploader("Upload JPG MRI File", type=["jpg"])
    if file_uploader:
        img_path = f"temp_img.{file_uploader.name.split('.')[-1]}"
        with open(file_uploader, 'wb') as file:
          file.write(file_uploader.read())
        st.text("Features are extracted...")
        img_features = image_feature_extractor(img_path)
    
        img_features.insert(0, 'Unnamed: 0', 0)
        st.text("Features are normalized...")
        scaled_img = scaler.transform(img_features)
        scaled_img_df = pd.DataFrame(scaled_img, columns=img_features.columns)
        selected_features_titles = ['Energy', 'Elongation', 'DifferenceAverage', 'DifferenceEntropy', 'DifferenceVariance', 'Id', 'Idm', 'Idmn', 'Imc1', 'Imc2', 'InverseVariance', 'JointAverage', 'JointEnergy', 'JointEntropy', 'MCC', 'MaximumProbability', 'SumAverage', 'SumEntropy', 'SumSquares', 'GrayLevelNonUniformityNormalized', 'HighGrayLevelRunEmphasis', 'LowGrayLevelRunEmphasis', 'Busyness', 'Coarseness', 'Complexity', 'DependenceEntropy', 'DependenceVariance', 'HighGrayLevelEmphasis', 'LowGrayLevelEmphasis', 'HighGrayLevelZoneEmphasis', 'LowGrayLevelZoneEmphasis', 'SizeZoneNonUniformity', 'SizeZoneNonUniformityNormalized', 'ZoneEntropy']

        best_features_img_df = scaled_img_df[selected_features_titles]
        if st.button("Predict"):
            result = classifier.predict(best_features_img_df)
    
            print(encoder.inverse_transform(result))
            st.success(f"The model predicts the scan to be: {result}")

main()
