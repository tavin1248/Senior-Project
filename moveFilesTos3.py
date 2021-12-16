# This file is used to convert the bmp images from the camera to jpg's
# Then upload the jpg file to the amazon s3 bucket.
# The file also keeps track of how many inputted images are in the directory
# when the number of inputted images goes up, a new photo has been taken
# and another jpg will be sent to the s3 bucket.

import boto3
from botocore.exceptions import NoCredentialsError
import os
import glob
import time
from PIL import Image

#Amazon s3 access keys
ACCESS_KEY = "AKIATNSKRGOXKMQMAV72"
SECRET_KEY = "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"
AWS_STORAGE_BUCKET_NAME = "senior-project-training-files"
AWS_S3_REGION_NAME = 'us-east-2'

#function to upload the images to the s3 bucket
def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

#path of the inputted image
path = 'C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/image001.jpg'
  
#printing the ctime of the last image that was added to the directory
c_time = os.path.getctime(path)
print("ctime since the epoch:", c_time)
  
#printing the ctime of the system
local_time = time.ctime(c_time)
print("ctime (Local time):", local_time)

# setting an array of bmp files
list_of_files = glob.glob('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/*.jpg')
count = 0
#this while loop monitors the size of the directory, if that size changes then a new photo will be added to the s3 bucket
# if the directory has over 50 bmp files, the ones that are the oldest will automatically be deleted
while True:
    list_of_files = glob.glob('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/*.bmp')

    # if the length of the directory changes, a new photo has been uploaded, therefore, that photo must be sent
    # to the s3 bucket so that the website can print the image
    if len(list_of_files) != count and len(list_of_files) > 0:
        list_of_files = glob.glob('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/*.bmp')
        img = max(list_of_files, key=os.path.getctime)

        photo = Image.open(img)
        new_img = photo.resize( (256, 256) )
        #converting the image from bmp --> png --> jpg
        new_img.save( 'C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/image001.png', 'png')
        im1 = Image.open('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/image001.png')
        im1.save('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/image001.jpg')

        img = 'C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/image001.jpg'
        uploaded = upload_to_aws(img, AWS_STORAGE_BUCKET_NAME, 'MachineLearning/UserTest/Image/image001.jpg')
        list_of_files = glob.glob('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/*.bmp')
        count = len(list_of_files)

    # deleting the oldest files if the dir is over 50 bmp's
    if len(list_of_files) > 50:
        list_of_files = glob.glob('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/*.bmp')
        img = min(list_of_files, key=os.path.getctime)
        os.remove(img)
        print('deleting oldest bmp')
        list_of_files = glob.glob('C:/Users/tardell/Senior_Project/MachineLearning/UserTest/Image/*.cfg')
        img = min(list_of_files, key=os.path.getctime)
        os.remove(img)
        print('deleting oldest cfg')
    
