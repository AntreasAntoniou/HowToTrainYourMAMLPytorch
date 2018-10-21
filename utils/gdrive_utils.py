import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def get_drive():
    try:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        return drive
    except:
        return None

def get_folder_id(folder_name, drive=None, f_id='root'):
    if drive == None:
        drive = get_drive()

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(f_id)}).GetList()
    for file1 in file_list:
        if file1['title'] == folder_name:
            return file1['id']

    return None


def delete_file(f_id, folder_name, drive=None):
    if drive == None:
        drive = get_drive()

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(f_id)}).GetList()
    for file1 in file_list:
        if file1['title'] == folder_name:
            file1.Delete()


def get_file(f_id, file_name, drive=None):
    if drive == None:
        drive = get_drive()

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(f_id)}).GetList()
    for file in file_list:
        # print(file['title'])
        if file['title'] == file_name:
            return file['id']

def download_dataset(dataset_name, path_to_save, drive=None):
    if drive == None:
        drive = get_drive()
    data_f_id = get_folder_id("data", drive=drive)

    print("Storing", data_f_id, "at", path_to_save)
    download_file_from_f_id(file_name=dataset_name, path_to_save=path_to_save, gdrive_folder=data_f_id, drive=drive)

def download_file_from_f_id(file_name, path_to_save, gdrive_folder, drive=None):

    if drive == None:
        drive = get_drive()
    try:
        folder_id = gdrive_folder
        file_id = get_file(f_id=folder_id, file_name=file_name, drive=drive)
        print('folder_id', folder_id, 'file_id', file_id)
        print("Downloading file {} in directory {}".format(file_name, path_to_save))
        f = drive.CreateFile({'id': file_id})
        f.GetContentFile(os.path.join(path_to_save, file_name))

    except:
        print("Failed to download")


def save_item_to_gdrive_f_id(file_to_save_path, gdrive_folder_id, drive=None):
    if drive == None:
        drive = get_drive()
    try:
        delete_file(f_id=gdrive_folder_id, folder_name=file_to_save_path.split("/")[-1], drive=drive)

        f = drive.CreateFile({"parents": [{"id": gdrive_folder_id}]})
        f.SetContentFile(file_to_save_path)
        f.Upload()
    except:
        print("Failed to upload file to GDRIVE")


def create_folder_under_folder_and_get_id(folder_name, f_id, drive=None):
    if drive == None:
        drive = get_drive()

    fold = drive.CreateFile({'title': folder_name,
                             "parents": [{"id": f_id}],
                             "mimeType": "application/vnd.google-apps.folder"})
    fold.Upload()
    foldertitle = fold['title']
    folderid = fold['id']
    print("Created folder {} which has id {}".format(foldertitle, folderid))
    return folderid


def create_folder_at_home_get_id(folder_name, drive=None):
    if drive == None:
        drive = get_drive()

    fold = drive.CreateFile({'title': folder_name,
                             "mimeType": "application/vnd.google-apps.folder"})
    fold.Upload()
    foldertitle = fold['title']
    folderid = fold['id']
    print("Created folder {} which has id {}".format(foldertitle, folderid))
    return folderid


def save_weights(file_to_save_path, gdrive_folder, drive=None):
    save_item_to_gdrive_f_id(
        file_to_save_path="{}.index".format(file_to_save_path),
        gdrive_folder_id=gdrive_folder, drive=drive)
    save_item_to_gdrive_f_id(
        file_to_save_path="{}.data-00000-of-00001".format(file_to_save_path),
        gdrive_folder_id=gdrive_folder, drive=drive)


def download_weights(file_to_save_path, gdrive_folder, drive=None):
    print("Download from folder id {} filename {}".format(gdrive_folder, file_to_save_path))
    download_file_from_f_id(
        file_to_save_path="{}.index".format(file_to_save_path),
        gdrive_folder=gdrive_folder, drive=drive)
    download_file_from_f_id(
        file_to_save_path="{}.data-00000-of-00001".format(file_to_save_path),
        gdrive_folder=gdrive_folder, drive=drive)

class ExperimentGoogleDriveSaver(object):
    def __init__(self, experiment_name, experiment_log_dir, experiment_sample_dir, experiment_saved_model_dir):

        self.drive = get_drive()
        self.exp_folder_id = get_folder_id(experiment_name)

        if self.exp_folder_id == None:
            self.exp_folder_id = create_folder_at_home_get_id(experiment_name)

        self.experiment_name = experiment_name
        self.experiment_log_dir = experiment_log_dir
        self.experiment_sample_dir = experiment_sample_dir
        self.experiment_saved_model_dir = experiment_saved_model_dir

        self.exp_log_folder_id = get_folder_id(folder_name="log", f_id=self.exp_folder_id)
        if self.exp_log_folder_id == None:
            self.exp_log_folder_id = create_folder_under_folder_and_get_id(folder_name="log", f_id=self.exp_folder_id)

        self.exp_sample_folder_id = get_folder_id(folder_name="sample_storage", f_id=self.exp_folder_id)
        if self.exp_sample_folder_id == None:
            self.exp_sample_folder_id = create_folder_under_folder_and_get_id(folder_name="sample_storage",
                                                                              f_id=self.exp_folder_id)

        self.exp_saved_model_folder_id = get_folder_id(folder_name="saved_model_storage", f_id=self.exp_folder_id)
        if self.exp_saved_model_folder_id == None:
            self.exp_saved_model_folder_id = create_folder_under_folder_and_get_id(folder_name="saved_model_storage",
                                                                                   f_id=self.exp_folder_id)

    def save_in_logs(self, file_to_save):
        save_item_to_gdrive_f_id(file_to_save_path=file_to_save, gdrive_folder_id=self.exp_log_folder_id,
                                 drive=self.drive)

    # def download_from_logs(self, filepath_to_download):
    #     download_file_from_gdrive_folder(file_to_save_path=filepath_to_download, gdrive_folder=self.exp_log_folder_id,
    #                                      drive=self.drive)

    def save_in_saved_models(self, file_to_save):
        save_weights(file_to_save_path=file_to_save, gdrive_folder=self.exp_saved_model_folder_id,
                     drive=self.drive)

    def download_from_saved_models(self, filepath_to_download):

        print("downloading weights to", filepath_to_download)
        download_weights(file_to_save_path=filepath_to_download,
                         gdrive_folder=self.exp_saved_model_folder_id,
                         drive=self.drive)

    def save_in_sample_storage(self, file_to_save):
        save_item_to_gdrive_f_id(file_to_save_path=file_to_save, gdrive_folder_id=self.exp_sample_folder_id,
                                 drive=self.drive)

    # def download_from_samples(self, filepath_to_download):
    #     download_file_from_gdrive_folder(file_to_save_path=filepath_to_download,
    #                                      gdrive_folder=self.exp_sample_folder_id,
    #                                      drive=self.drive)

    def save_in_tensorboard_objects(self, file_to_save, phase="train"):

        folder_path = "{}".format(file_to_save, phase)
        print(folder_path)

    # def download_from_tensorboard_objects(self, filepath_to_download, phase="train"):
    #     download_file_from_gdrive_folder(file_to_save_path=filepath_to_download,
    #                                      gdrive_folder=self.exp_sample_folder_id,
    #                                      drive=self.drive)
