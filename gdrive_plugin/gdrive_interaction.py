#
# Interact with the GDRIVE files remotely
#
# Author: Stijn Van Hoey
# INBO

from __future__ import print_function
import argparse
import os
import sys
import re
import io

import httplib2
from oauth2client.file import Storage
from apiclient import discovery
from apiclient.http import MediaIoBaseDownload

"""
! ATTENTION !
Make sure to enable the google API by following
[this](https://developers.google.com/drive/v3/web/quickstart/python) tutorial.
! However, make sure to adapt the scope to
SCOPES = 'https://www.googleapis.com/auth/drive.readonly', hence use the credentials_setup.py file
included instead of the quistart.py from the tutorial
The file will create a credentials file from which following authentifications
can be used.
"""

class GDriveConnect(object):
    """
    This class provides the tools to interact with googe drive and look for files and their corresponding URI.
    """

    def __init__(self, cred_file,
                 folder_hash="0B4xlTsZWnBR9X3gwazAwWEdYcGM"):
        """
        Within the given drive folder, search for files, retrieve the
        corresponding URI for the files and download the files. Besides the `search_file` functionality, the more general query options for G-drive are applicable as well.

        Parameters
        -----------
        cred_file : str
            local json file containing the credentials for the API access to the gdrive. The file can be retrieved by running the `gdrive_account_setup.py` after creating a API token
            in your google account.
        folder_hash : str
            hash of the Gdrive location to search for files. The easiest way to get these is by
            checking the last section of the URL when opening the drive.
        """
        self.cred_file_location = cred_file
        self.drive_id = folder_hash
        self.service = self._connection()

    def _connection(self):
        """using an credentials file, setup the gdrive connection as a service
        """
        storage = Storage(self.cred_file_location)
        credentials = storage.get()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('drive', 'v3', http=http)
        return service

    @staticmethod
    def get_gdrive_url(file_id):
        """transform the file hash to the corresponding URL to open the file
        """
        return "".join(["https://drive.google.com/open?id=", file_id])

    def list_deployments(self):
        """lists the deployment folders in the drive folder

        Returns
        -------
        deployment_folder: generator returning the deployment_folder dicts
        """
        folders = "mimeType = 'application/vnd.google-apps.folder'"
        parent = "'{}' in parents".format(self.drive_id)
        query = " and ".join([folders, parent])

        page_token = None
        while True:
            response = self.service.files().list(q=query,
                                                 spaces='drive',
                                                 pageToken=page_token).execute()
            for deployment_folder in response.get('files', []):
                yield deployment_folder

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break;

    def list_deployment_images(self, deployment_folder_hash):
        """lists the images in a deployment folder
        """
        folders = "mimeType='image/jpeg'"
        parent = "'{}' in parents".format(deployment_folder_hash)
        query = " and ".join([folders, parent])

        page_token = None
        while True:
            response = self.service.files().list(q=query,
                                                 spaces='drive',
                                                 pageToken=page_token).execute()
            for image in response.get('files', []):
                yield image
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break;

    def download_image(self, image, pre_path="./"):
        """download an image using the corresponding file metadata
        """
        image_hash = image["id"]
        image_name = image["name"]

        request = self.service.files().get_media(fileId=image_hash)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download {}%.".format(int(status.progress() * 100)))

        with open(os.path.join(pre_path, image_name), "wb") as img:
            img.write(fh.getvalue())
