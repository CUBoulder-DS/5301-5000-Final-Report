# %% [markdown]
# First - install dependencies in your local machine or python environment (Python 3.8):
# ```
# pip install torch #reference torch documentation for correct install commands for your system: https://pytorch.org/get-started/locally/
# pip install torch_directml #necessary if you have windows with windows sub-system for linux (WSL) to achieve a degree of optimization in Python 3.8
# pip install numpy #you probably already have it
# pip install pandas #you probably already have it
# pip install dlib #easiest to handle on linux / mac systems - in Windows it's a nightmare and will take me about 8 hrs to document.
# pip install deepface 
# ```
# 
# Here, we've taken the script from the FairFace GitHub and converted it to a class.  We did this out of necessity, as we sought the ability to evaluate one image at a time, or images in a batch, as well as the ability to classify an image with or without pre-processing (and optionally, without storing the pre-processed image).  Without a class, it would have required re-loading the detection and classification models to CPU/GPU with every function call.  As such, we built this class which gave us the flexibility we sought in processing our data.
# 
# This class can be used as part of the script, or could potentially be re-uploaded back to the FairFace GitHub, potentially advancing its design so that it could eventually become a pip package installable similar to DeepFace for ease of use by others.
# 
# To be able to use this class and script, you must first download or clone the FairFace package from GitHub directly (not currently a pip package).  You can do that here:
# https://github.com/dchen236/FairFace
# 
# After you've pulled the file, you'll need to download copies of the models and save them in a new folder called fair_face_models.  Models are located here: https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu
# 
# After you do these steps, you can store this script in the main folder where you've saved/copied the FairFace repository. 
# 
# From there, I recommend you get access to/download any images you wish to process and store them in a local folder, and use some code to generate a CSV of all the files in that folder.
# 

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
        """_summary_
        return instance of the FairFace class with models loaded to CPU/GPU
        leverages torch_directml for users with AMD GPU systems (especially on Linux)
        """
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
        """_summary_
        loads the appropriate FairFace Detection model to the appropriate device (CPU/GPU)
        """
        self.__model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt',map_location=self.__map_location))
        self.__model_fair_7 = self.__model_fair_7.to(self.__device)
        self.__model_fair_7.eval()        
        self.__model_fair_4.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_4_20190809.pt',map_location=self.__map_location))
        self.__model_fair_4 = self.__model_fair_4.to(self.__device)
        self.__model_fair_4.eval()

    def set_save_location(self,path):
        """_summary_
        not sure this function is needed.
        Args:
            path (_type_): string - path to save location
        """
        self.__save_detections_at = path
        self.__ensure_dir(path)

    def __ensure_dir(self,directory):
        """_summary_
        This function may not be needed.
        Args:
            directory (_type_): path to a folder / location to ensure it exists
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def detect_face(self, img_path, default_max_size = 800, size=300, padding=0.25,
                    save=False):
        """detects faces in an image; serves as a helper function to analyze function
            if enforce_detection is enabled.
            returns an image if save is False, otherwise saves the image at self.__save_location
            for later processing
        Args:
            img_path (string): filepath to an image in which to detect faces
            default_max_size (int, optional): Defaults to 800.
            size (int, optional):  Defaults to 300.
            padding (float, optional):  Defaults to 0.25.
            save (bool, optional): If true, will save a .jpg file of the input file at self.__save_location. Defaults to False.

        Returns:
            np.ndarray: an opened, pre-processed/cropped image focused on a face for categorization models.
            none (if save is enabled) -> saves the np.ndarray to the specified path in this instance's 
            self.__save_detections_at variable.
        """
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
        """evaluates an input image on either the fair7 or fair4 model. defaults to fair7.
enables analysis and classification with or without preprocessing (defaults to preprocessing enabled).
if pre-processing is enabled, calls to detect_face and gets its return data for classifying the image. 

        Args:
            image (str): relative filepath to an image from the current directory
            mode (str, optional): options include 'fair4' and 'fair7' for the respective FairFace models. Defaults to 'fair7'.
            enforce_detection (bool, optional): Allows this function and the selected model to evaluate an image directly without preprocessing. Defaults to True.

        Returns:
            result (pandas DataFrame): a dataframe with the respective predictions and calculations from the selected FairFace model on the provided image.
        """
        race_scores,gender_scores,age_scores=[],[],[]
        race_pred,gender_pred,age_pred = [],[],[]
        face_names = []
        face_names.append(image)
        if enforce_detection:
            image = self.__trans(self.detect_face(image)[0])
        else:
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
        if mode == 'fair7':
            race_src = [result['race_preds_fair']==0,result['race_preds_fair']==1,
                        result['race_preds_fair']==2,result['race_preds_fair']==3,
                        result['race_preds_fair']==4,result['race_preds_fair']==5,
                        result['race_preds_fair']==6]
            race_dst = ['White','Black','Latino_Hispanic',
                        'East Asian','Southeast Asian','Indian',
                        'Middle Eastern']
        elif mode == 'fair4':
            race_src = [result['race_preds_fair']==0,result['race_preds_fair']==1,
                        result['race_preds_fair']==2,result['race_preds_fair']==3]
            race_dst = ['White','Black','Asian','Indian']
        else:
            print("unsupported mode.")

        gen_src = [result['gender_preds_fair']==0,result['gender_preds_fair']==1]
        gen_dst = ['Male','Female']
        age_src = [result['age_preds_fair']==0,result['age_preds_fair']==1,
                   result['age_preds_fair']==2,result['age_preds_fair']==3,
                   result['age_preds_fair']==4,result['age_preds_fair']==5,
                   result['age_preds_fair']==6,result['age_preds_fair']==7,
                   result['age_preds_fair']==8]
        age_dst = ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','70+']
        result['race_preds_fair']=np.select(race_src,race_dst,result['race_preds_fair'])
        result['gender_preds_fair']=np.select(gen_src,gen_dst,result['gender_preds_fair'])
        result['age_preds_fair']=np.select(age_src,age_dst,result['age_preds_fair'])

        return result[['face_name_align',
                'race_preds_fair',
                'gender_preds_fair',
                'age_preds_fair',
                'race_scores_fair',
                'gender_scores_fair',
                'age_scores_fair']]

    def get_map_loc(self):
        """debugging function to ensure the correct available device(s) (cpu,gpu) are available and used

        Returns:
            torch.device: either cpu or cuda:0
        """
        return self.__map_location

    def get_device(self):
        """debugging function to ensure the correct available device(s) (cpu,gpu) are available and used

        Returns:
            torch.device: either torch_directml device, or cuda:0
        """
        return self.__device

    def batch_detect_analyze(self, img_paths):
        #function not currently used and can be improved upon.
        for i, img in enumerate(img_paths):
            self.detect_face(img)
        img_names = [os.path.join(img_paths, x) for x in os.listdir(img_paths)]
        # for img in img_names:
        #     self.analyze(self.__save_detections_at+'/'+img)
        # return 0

    def _rect_to_bb(self,rect):
        """function was present and undocumented in original script
        code was not used elsewhere within the source script

        Args:
            rect (_type_): _description_

        Returns:
            _type_: _description_
        """
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

def normalize_output(model,data):
    """
    format the output from batch_analyze in a standard fashion for comparison
    match the master dataframe format for the final product

    Args:
        model (str): _description_
        data (pandas dataframe): output from batch_analyze function

    Returns:
        data (pandas dataframe): includes the following normalized columns:
            model: FairFace|DeepFace (str)
            pred_age_grp (str): a bin of age ranges (i.e. "10-19", "20-29", etc in which the model predicts the subject lay)
            pred_age_lower (int): lower bound of the predicted age bin
            pred_age_upper (int): upper bound of the predicted age bin
            pred_gender (str): Male|Female
        individual categorization scores are omitted from each models' output.
    """
    if model == 'FairFace':       
        #normalize race to those output by UTK
        data.columns=['file','pred_race','pred_gender','pred_age_grp','race_scores','gender_scores','age_scores']
        data=data.drop(labels=['race_scores','gender_scores','age_scores'],axis=1)
        race_src = [data['pred_race']=='East Asian',data['pred_race']=='Southeast Asian',
                    data['pred_race']=='Latino_Hispanic',data['pred_race']=='White',
                    data['pred_race']=='Black',data['pred_race']=='Indian',data['pred_race']=='Middle Eastern']
        race_dest = ['Asian','Asian','Other','White','Black','Indian','Other']
        data['pred_race'] = np.select(race_src,race_dest,data['pred_race'])
        data['pred_age_grp'] = np.where(data['pred_age_grp']=='70+','70-130',data['pred_age_grp'])

    else:
        #normalize race to those output by UTK
        data.columns=['age','pred_gender','pred_race','file']
        #asian, white, middle eastern, indian, latino and black
        race_src = [data['pred_race']=='asian',data['pred_race']=='white',data['pred_race']=='middle eastern',
                    data['pred_race']=='indian',data['pred_race']=='latino',data['pred_race']=='black']
        race_dest = ['Asian','White','Other','Indian','Other','Black']
        data['pred_race'] = np.select(race_src,race_dest,data['pred_race'])
        #remap predicted genders
        gen_src = [data['pred_gender']=='Man',data['pred_gender']=='Woman']
        gen_dst = ['Male','Female']
        data['pred_gender'] = np.select(gen_src,gen_dst,data['pred_gender'])
        #bin the ages according to predicted age
        bins = [0, 3, 10, 20, 30, 40, 50, 60, 70, np.inf]
        group_names = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-130"]
        data["pred_age_grp"] = pd.cut(data['age'], bins, right=False, labels=group_names)
        data = data.drop(labels=['age'],axis=1)

    data[["pred_age_lower","pred_age_upper"]] = data['pred_age_grp'].str.split('-',expand=True).astype(int) #convert to int somehow
    
    data = data[['file','pred_race','pred_gender','pred_age_grp','pred_age_lower','pred_age_upper']]
    return data


def batch_analyze(model, in_csv, out_csv, preprocess=True):
    """allows user to select a facial recognition/categorization model 
    and a CSV of image file locations to evaluate all those images against the selected model,
    and output the results to a CSV.

    Note:  The column with paths to the image files' directory are relative paths, and the column in the 
    CSV containing those paths must be labeled 'img_path'

    Args:
        model (str): 'FairFace' or 'DeepFace' - model to use when evaluating images
        in_csv (str): relative path location to a CSV containing paths to images for evaluation
        out_csv (_type_): relative path location and name in which to store output for this evaluation
        preprocess (bool, optional): if true, facial detection is performed on each image within the given model before it performs image classification. Defaults to True.
    """
    files = pd.read_csv(os.path.join(in_csv))['img_path']
    result=[]
    if model == 'FairFace':
        FairFace=fairface()
        for index,record in enumerate(files):
            if index % 100 == 0:
                print('{}/{}'.format(index,len(files)))
            curr = FairFace.analyze(os.path.join(record),mode='fair7',enforce_detection=preprocess)
            if len(result)==0:
                result = curr
            else:
                result = pd.concat([result,curr])
    else:
        from deepface import DeepFace
        import time
        #probably need to introduce a cool-down period
        if preprocess=='True':
            backend=input('select DeepFace Detector Backend: ')
        for index,record in enumerate(files):
            if index % 100 == 0:
                print('{}/{}'.format(index,len(files)))
                if index > 0:
                    print('sleeping / cooldown for 45 seconds')
                    time.sleep(45)

            try:
                if preprocess=='False':
                    curr = DeepFace.analyze(img_path=record,enforce_detection=False,actions=['age','gender','race'],silent=True)
                else:
                    curr = DeepFace.analyze(img_path=record,enforce_detection=True,actions=['age','gender','race'],silent=True,detector_backend=backend)
                curr=curr[0]
                del curr['gender']
                del curr['race']
                del curr['region']
                curr['file'] = record
                curr = pd.DataFrame(curr,index=[0])
                if len(result) == 0:
                    result = pd.DataFrame(curr)
                else:
                    result = pd.concat([result,curr])
            except:
                print("error processing {}".format(record))
    result.to_csv(os.path.join('non_normalized_'+model+'_'+out_csv))
    result=normalize_output(model,result)
    result['pred_model'] = model
    result.to_csv(os.path.join(out_csv))


# %% [markdown]
# The below section allows this to be run via command line as follows:
# 
# `python MasterScript.py --in_csv [input csv file path] --out_csv [desired output csv file path and name] --model [DeepFace|FairFace] --preproc [True|False]`
# 
# in_csv parameter allows you to specify an input CSV.  The csv must have, at a minimum, a column with the name img_path that specifies the relative path to the files you'll be working with.
# 
# out_csv parameter allows you to specify a file to which you wish to write your output.  This is also written with respect to the current directory.
# 
# model parameter allows you to select whether you're using DeepFace or FairFace.  If DeepFace is selected, you will be prompted to enter the facial detection backend once (recommend using mtcnn or opencv, others are failing to install or showing substantial failure to detect faces)
# 
# preproc parameter allows you to specify whether or not you wish for the image to have a face detected and pre-processed prior to attempted classification.  Setting to True means that faces will be detected first, and False skips detection and goes straight to classification
# 
# Also this notebook shouldn't be used directly - it is simply amplifying documentation to better explain the script file MasterScript.py.

# %%
#turns this into a command-line script so that it can be called as follows:
from glob import glob
"python MasterScript.py --in_csv [input csv file path] --out_csv [desired output csv file path and name] --model [DeepFace|FairFace] --preproc [True|False]"
if __name__ == '__main__':
    example_text = '''
examples:
    #generate a CSV for iteration from jpg files in folder 'path3' in the current directory & save as UTKpart3.csv
    python MasterScript.py --fp part3 --out_csv UTKpart3.csv 
    python MasterScript.py -f part3 -o UTKpart3.csv

    #evalutate contents of UTKpart3.csv using FairFace without pre-processing and save output to FF_UTKpart3_no_preproc.csv
    python MasterScript.py --in_csv UTKpart3.csv --model FairFace --preproc False --out_csv FF_UTKpart3_no_preproc.csv
    python MasterScript.py -i UTKpart3.csv -m FairFace -p False -o FF_UTKpart3_no_preproc.csv
    
    #evaulate contents of UTKpart3.csv using DeepFace with preprocessing enabled (requires specifying facial detection backend of mtcnn or opencv for DeepFace)
    # and save output to DF_UTKpart3_preproc[mtcnn|opencv].csv
    python MasterScript.py --in_csv UTKpart3.csv --model DeepFace --preproc True --out_Csv DF_UTKpart3_preproc[mtcnn|opencv].csv
    python MasterScript.py -i UTKpart3.csv -m DeepFace -p True -o DF_UTKpart3_preproc[mtcnn|opencv].csv
'''
    parser = argparse.ArgumentParser(
        prog='python MasterScript.py',
        description='script to iterate through images using FairFace or DeepFace',
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    #in_csv arg
    parser.add_argument('-i','--in_csv',dest='in_csv',action='store',
                        help='csv containing filepaths to input images to categorize'
                        )
    #out_csv arg
    parser.add_argument('-o','--out_csv',dest='out_csv',action='store',
                        help='filename in which to store model output'
                        )
    #model arg
    parser.add_argument('-m','--model',dest='model',action='store',
                        help='FairFace or DeepFace'
                        )
    #preprocess arg
    parser.add_argument('-p','--preproc',dest='preproc',action='store',
                        help='True or False - preprocess images before evaluating'
                        )
    #fp arg - allows you to generate an input CSV file to use within this script
    #depends on having a folder in the same root that contains .jpg files.
    parser.add_argument('-f','--fp',dest='PATH',action='store',
                    help='folder containing images for processing'
                    )
    args=parser.parse_args()
    if args.PATH is None:
        batch_analyze(args.model,args.in_csv,args.out_csv,args.preproc)
    else:
        EXT = "*.jpg"
        all_jpg_path = [file 
                for path,subdir, files in os.walk(args.PATH)
                for file in glob(os.path.join(path,EXT))]
        all_jpg = [os.path.basename(file)
            for path, subdirs, files in os.walk(args.PATH)
            for file in glob(os.path.join(path,EXT))
        ]
        present_files = pd.DataFrame({'img_paths':all_jpg_path,'file':all_jpg})
        present_files.to_csv(os.path.join(args.out_csv),index=False)
    