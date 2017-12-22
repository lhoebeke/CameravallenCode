# 
# Downloading deployment images from google drive to a local storage device 
#
# S. Van Hoey
# 

import sys
import click

from gdrive_interaction import GDriveConnect


@click.command()
@click.argument('credentials', type=click.Path(exists=True))
@click.argument('local_dir', type=click.Path(exists=True))
@click.option('--gdrive_hash', default="0B4xlTsZWnBR9X3gwazAwWEdYcGM", 
              help='Hash (see URL) of the google drive folder to download')
@click.option('--nlim', default=None, 
              help='Limit the number of files to check for each deployment')
def download_gdrive(credentials, gdrive_hash, local_dir, nlim):
    """
    Download all deployments from a google drive folder, using the 
    google API CREDENTIALS file for the connection and a LOCAL_DIR
    to copy data to
    """
    gconnect = GDriveConnect(cred_file=credentials,
                             folder_hash=gdrive_hash)

    deployments = list(gconnect.list_deployments())
    for deployment in deployments:
        click.echo("Handling deployment {}...".format(deployment["name"]))
        gconnect.download_deployment_images(deployment, 
                                            local_dir, 
                                            nlim=nlim)

if __name__ == "__main__":
    sys.exit(download_gdrive())