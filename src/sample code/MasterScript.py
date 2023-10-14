# %%
from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch_directml
import dlib
import os
import argparse


class fairface:

    def __init__(self,**kwargs):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
        self.sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
        self.base = 2000
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else torch_directml.device())
        self.__map_location= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model_fair_7 = torchvision.models.resnet34(pretrained=True)
        self.__model_fair_7.fc = nn.Linear(self.__model_fair_7.fc.in_features, 18)
        self.__model_fair_7 = self.__model_fair_7.to(self.__device)
        self.__model_fair_7.eval()
        self.__model_fair_4 = torchvision.models.resnet34(pretrained=True)
        self.__model_fair_4.fc = nn.Linear(self.__model_fair_4.fc.in_features, 18)
        self.__model_fair_4 = self.__model_fair_4.to(self.__device)
        self.__model_fair_4.eval()

        self.__save_detections_at = ""
        self.__trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.__load_state_dicts()

    def __load_state_dicts(self):
        self.__model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt',map_location=self.__map_location))
        self.__model_fair_7 = self.__model_fair_7.to(self.__device)
        self.__model_fair_7.eval()        
        self.__model_fair_4.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_4_20190809.pt',map_location=self.__map_location))
        self.__model_fair_4 = self.__model_fair_4.to(self.__device)
        self.__model_fair_4.eval()

    def set_save_location(self,path):
        self.__save_detections_at = path
        self.__ensure_dir(path)

    def __ensure_dir(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def detect_face(self, img_path, default_max_size = 800, size=300, padding=0.25,
                    save=False):
        #print(os.path.join(img_path))
        img = dlib.load_rgb_image(os.path.join(img_path))
        old_height, old_width, _ = img.shape
        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)
        dets = self.cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(img_path))
        # Find the 5 face landmarks we need to do the alignment.
        else:
            faces = dlib.full_object_detections()
            for detection in dets:
                rect = detection.rect
                faces.append(self.sp(img, rect))
            images = dlib.get_face_chips(img, faces, size=size, padding = padding)
            if save: 
                for idx, image in enumerate(images):
                    img_name = img_path.split("/")[-1]
                    path_sp = img_name.split(".")
                    face_name = os.path.join(self.__save_detections_at,  path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
                    dlib.save_image(image, face_name)
                print("Saved detected face(s) at "+self.__save_detections_at)
            else:
                return images

    def analyze(self,image,mode='fair7',enforce_detection=True):
        indices={'race_start':0,'race_end':0,'gen_start':0,'gen_end':0,'age':0}
        race_scores,gender_scores,age_scores=[],[],[]
        race_pred,gender_pred,age_pred = [],[],[]
        face_names = []
        face_names.append(image)
        race_dict={}
        gender_dict = {0:'Male',1:'Female'}
        age_dict = {
            0:'0-2',
            1:'3-9',
            2:'10-19',
            3:'20-29',
            4:'30-39',
            5:'40-49',
            6:'50-59',
            7:'60-69',
            8:'70+'
        }
        if mode == 'fair7':
            #print('fair7')
            #outputs = self.__model_fair_7(image)
            race_dict = {
                0:'White',
                1:'Black',
                2:'Latino_Hispanic',
                3:'East Asian',
                4:'Southeast Asian',
                5:'Indian',
                6:'Middle Eastern'
            }
            indices={'race_start':0,'race_end':7,'gen_start':7,'gen_end':9,'age_start':9,'age_end':18}
        elif mode == 'fair4':
            #print('fair4')
            #outputs = self.__model_fair_4(image)
            race_dict = {
                0:'White',
                1:'Black',
                2:'Asian',
                3:'Indian',
            }
            indices={}
        else:
            print("unsupported mode.")

        if enforce_detection:
            image = self.__trans(self.detect_face(image)[0])
        else:
            #print(os.path.join(image))
            image = self.__trans(dlib.load_rgb_image(os.path.join(image)))
        
        image=image.view(1,3,224,224)
        image=image.to(self.__device)

        outputs = None

        if mode=='fair7':
            outputs = self.__model_fair_7(image)
        else:
            outputs = self.__model_fair_4(image)
        
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)
        race_outputs,gender_outputs,age_outputs=[],[],[]
        race_score,gender_score,age_score=0,0,0
        #print("Outputs: ", outputs)
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]
        if mode=='fair7':
            race_outputs = outputs[:7]
            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        else:
            race_outputs = outputs[:4]
            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))

        race_pred = np.argmax(race_score)
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))
        gender_pred = np.argmax(gender_score)
        age_pred=np.argmax(age_score)

        race_scores.append(race_score)
        age_scores.append(age_score)
        gender_scores.append(gender_score)
        #print(face_names,race_pred,gender_pred,age_pred,race_scores,gender_scores,age_scores)
        result = pd.DataFrame(
               [face_names,
                [race_pred],
                [gender_pred],
                [age_pred],
                race_scores, 
                gender_scores,
                age_scores] 
        ).T
        result.columns = ['face_name_align',
                'race_preds_fair',
                'gender_preds_fair',
                'age_preds_fair',
                'race_scores_fair',
                'gender_scores_fair',
                'age_scores_fair']
        #result[['race','gender','age']] = '','',''
        #print(result)
        result['race_preds_fair'] = result.apply(lambda row: race_dict[row['race_preds_fair']],axis=1)
        result['gender_preds_fair']=result.apply(lambda row: gender_dict[row['gender_preds_fair']],axis=1)
        result['age_preds_fair']=result.apply(lambda row: age_dict[row['age_preds_fair']],axis=1)
        
        return result[['face_name_align',
                'race_preds_fair',
                'gender_preds_fair',
                'age_preds_fair',
                'race_scores_fair',
                'gender_scores_fair',
                'age_scores_fair']]

    def get_map_loc(self):
        return self.__map_location

    def get_device(self):
        return self.__device

    def batch_detect_analyze(self, img_paths):
        for i, img in enumerate(img_paths):
            self.detect_face(img)
        img_names = [os.path.join(img_paths, x) for x in os.listdir(img_paths)]
        for img in img_names:
            self.analyze_face(self.__save_detections_at+'/'+img)
        return 0

    def _rect_to_bb(self,rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)        
    




# %%
FairFace = fairface()

# %%
#FairFace.analyze('./19/112/104662-1.jpg',mode='fair7',enforce_detection=True)
#FairFace.detect_face('./19/112/100584-0.jpg')

# %%
#FairFace.analyze('./19/112/100057-1.jpg',mode='fair7',enforce_detection=False)

# %%
#FairFace.analyze('./19/112/100057-1.jpg',mode='fair7',enforce_detection=True)
from deepface import DeepFace

# %%

# def create_image_csv(target_folder):
#     pass

def batch_analyze(model, in_csv, out_csv, preprocess=True):
    files = pd.read_csv(os.path.join(in_csv))['img_path']
    result=[]
    if model == 'FairFace':
        for index,record in enumerate(files):
            #print(record)
            curr = FairFace.analyze(os.path.join(record),mode='fair7',enforce_detection=preprocess)
            if len(result)==0:
                result = curr
            else:
                result = pd.concat([result,curr])
    else:
        backend=input('select DeepFace Detector Backend: ')
        #cat_model=input('select DeepFace model: ')
        for index,record in enumerate(files):
            #DeepFace.analyze(img_path=image_path,enforce_detection=False,silent=True)[0]
            print(record)

            try:
                curr = DeepFace.analyze(img_path=record,enforce_detection=preprocess,actions=['age','gender','race'],silent=True,detector_backend=backend)
                #print("Post-analysis:",curr)
                curr=curr[0]
                del curr['gender']
                del curr['race']
                del curr['region']
                print(curr)
                curr = pd.DataFrame(curr,index=[0])
                #print(curr)
                #print(result)
                if len(result) == 0:
                    result = pd.DataFrame(curr)
                else:
                    result = pd.concat([result,curr])
                #print(result,'\n\n\n\n\n')
            except:
                print("error processing {}".format(record))
    
    result.to_csv(os.path.join(out_csv))
            
#type(FairFace.analyze('./19/112/100057-1.jpg',mode='fair7',enforce_detection=False))
#batch_analyze('FairFace',in_csv='AFAD_SHORT.csv',out_csv='test2.csv')


# %%
#batch_analyze('FairFace',in_csv='AFAD_SHORT.csv',out_csv='test_ipynb.csv')

# %%
#batch_analyze('FairFace',in_csv='AFAD_SHORT.csv',out_csv='test2_nopreproccess.csv',preprocess=False)

# %%
#batch_analyze('DeepFace',in_csv='AFAD_SHORT.csv',out_csv='DFtest.csv',preprocess=False)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #in_csv arg
    parser.add_argument('--in_csv',dest='in_csv',action='store',
                        help='csv containing filepaths to input images to categorize'
                        )
    #out_csv arg
    parser.add_argument('--out_csv',dest='out_csv',action='store',
                        help='filename in which to store model output'
                        )
    #model arg
    parser.add_argument('--model',dest='model',action='store',
                        help='FairFace or DeepFace'
                        )
    #preprocess arg
    parser.add_argument('--preproc',dest='preproc',action='store',
                        help='True or False - preprocess images before evaluating'
                        )
    args=parser.parse_args()
    batch_analyze(args.model,args.in_csv,args.out_csv,args.preproc)
    


