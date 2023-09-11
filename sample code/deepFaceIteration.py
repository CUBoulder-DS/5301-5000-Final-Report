from deepface import DeepFace

import os, os.path, pandas as pd, numpy as np, argparse

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
]

def detect_predict_face(img_paths, save_loc):
    result = pd.DataFrame(columns=['filename','race','age','gender'])
    result_detail = pd.DataFrame(columns=[
        'emotion','dominant_emotion','region','age','gender',
        'dominant_gender','race','dominant_race'
    ])

    for index, image_path in enumerate(img_paths):
        if index % 1000 == 0:
            print('---%d/%d---' %(index, len(img_paths)))
            try:
                curr_data = pd.read_csv('deepfaceout.csv')
                curr_data = pd.concat([curr_data,result])
                curr_data.to_csv('deepfaceout.csv')
            except FileNotFoundError:
                result.to_csv('deepfaceout.csv')
            except KeyboardInterrupt:
                break
            try:
                curr_data = pd.read_csv('deepfaceout_detail.csv')
                curr_data = pd.concat([curr_data,result_detail])
                curr_data.to_csv('deepfaceout_detail.csv')
            except FileNotFoundError:
                result.to_csv('deepfaceout_detail.csv')
            except KeyboardInterrupt:
                break
        # try:
        #     det_face = DeepFace.extract_faces(img_path=image_path)
        # except:
        #     print("Sorry, there were no faces found in '{}'".format(image_path))
        #     continue
        try:
            curr_face_data =  DeepFace.analyze(img_path=image_path,enforce_detection=False,silent=True)[0]
            #print(curr_face_data)
            row = pd.DataFrame(
                {
                    'age':curr_face_data['age'],
                    'race':curr_face_data['dominant_race'],
                    'gender':curr_face_data['dominant_gender'],
                    'filename':image_path
                },index=[0] 
            )
            row_detail = pd.DataFrame(
                {
                    'age':curr_face_data['age'],
                    'dominant_race':curr_face_data['dominant_race'],
                    'dominant_gender':curr_face_data['dominant_gender'],
                    'emotion':curr_face_data['emotion'],
                    'dominant_emotion':curr_face_data['dominant_emotion'],
                    'region':curr_face_data['region'],
                    'gender':curr_face_data['gender'],
                    'race':curr_face_data['race'],
                    'filename':image_path                
                }
            )
            #print(row)s
            result = pd.concat([result,row])
            result_detail = pd.concat([result_detail,row_detail])
        except:
            print("Error processing file %s. Continuing iteration...")
            continue 
    result.to_csv('deepfaceout.csv')
    result_detail.to_csv('deepfaceoutdetail.csv')
    #print(result)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    #Please create a csv with one column 'img_path', contains the full paths of all images to be analyzed.
    #Also please change working directory to this file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', dest='input_csv', action='store',
                        help='csv file of image path where col name for image path is "img_path')
    args = parser.parse_args()
    SAVE_DETECTED_AT = "detected_faces"
    ensure_dir(SAVE_DETECTED_AT)
    imgs = pd.read_csv(args.input_csv)['img_path']
    detect_predict_face(imgs, SAVE_DETECTED_AT)
    print("detected faces are saved at ", SAVE_DETECTED_AT)