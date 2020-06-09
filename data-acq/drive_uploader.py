'''
Automation to upload to Google Drive
when there is new file in DATA_PATH
using pydrive library

To do:
    - Initialize uploaded files to files existed in drive
    - If upload success and verified, delete file locally
'''

import os
from time import sleep
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from datetime import datetime

#### Authentication and drive initialization ####
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


# Folder to check
DATA_PATH = 'C:/Users/rss75/Documents/GitHub/ta-shop/data-acq/accel-data/'

# New uploaded files
uploaded_set = set()

def retrieve_files(path):
    # Define empty set of files
    retrieved_files = set()

    # Iterate to all files in folder
    for file in os.listdir(path):
        fullpath = os.path.join(path, file)
        
        # If is file, add to set of files
        if os.path.isfile(fullpath):
            retrieved_files.add(file)

    return retrieved_files


def check_new_files(path):
    retrieved_set = set()
    for filename in retrieve_files(path):
        retrieved_set.add(filename)

    # Check difference between uploaded set and retrieved set
    new_set = retrieved_set - uploaded_set

    return new_set


def check_new_files_metadata(path):
    retrieved_set = set()
    for filename in retrieve_files(path):
        # Extract file info (metadata?)
        stat = os.stat(os.path.join(path, filename))

        # Extract time and size metadata of file
        time = stat.st_ctime
        size = stat.st_size

        retrieved_set.add(filename)

    # Check difference between uploaded set and retrieved set
    new_set = retrieved_set - uploaded_set

    return new_set


def get_drive_structure():
    fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file in fileList:
        print('Title: %s, ID: %s' % (file['title'], file['id']))
        # Get the folder ID that you want
        if(file['title'] == "To Share"):
            fileID = file['id']


def upload_new_files(path, parent_id):
    print('Checking new files...',path)
    new_files = check_new_files(path)

    # If there are new files to upload
    if new_files:
        print('Uploading',len(new_files),'new files to',parent_id)
        for file in check_new_files(path):
            try:
                file_path = os.path.join(path, file)

                # Add upload date 
                filename = file[:-12] + str(datetime.now().strftime("%y_%m_%d_")) + file[-12:]

                # Upload to folder with specific id
                f = drive.CreateFile({
                        'title': filename,
                        'parents': [{
                            'id': parent_id
                        }]
                    })
                f.SetContentFile(file_path)
                f.Upload()

                # Weird bug fix to prevent memleak by preventing deletion
                f = None

                print('--- Upload',filename,'successful, removing from local storage...')
                
                # Upload success, add file to uploaded set
                uploaded_set.add(file)

                # !!! Delete file if upload success
                if os.path.exists(file_path):    
                    os.remove(file_path)
                else:
                    print('(!) File does not exist:',file_path)
            
            except:
                print('(!) Error occured for',file,', retrying in next upload batch')


    # If no new files to upload
    else:
        print('No new files to upload, retrying in 5 seconds...')
        sleep(5)


if __name__ == '__main__':
    # Infinite loop
    while True:
        # Upload to specific folder ('Tugas Akhir/data/accel-data-cont')
        upload_new_files(DATA_PATH, '1ysdKEH2LBLRHV0miaNgbjFECV11uH5JK')