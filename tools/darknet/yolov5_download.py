import os
from pathlib import Path
import torch

# from https://github.com/ultralytics/yolov5

def attempt_download(name='yolov5s.pt'):
    file = Path(name).name
    msg = name + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']  # available models

    if file in models and not os.path.isfile(name):
        try:  # GitHub
            url = 'https://github.com/ultralytics/yolov5/releases/download/v3.0/' + file
            print('Downloading %s to %s...' % (url, name))
            torch.hub.download_url_to_file(url, name)
            assert os.path.exists(name) and os.path.getsize(name) > 1E6  # check
        except Exception as e:  # GCP
            print('Download error: %s' % e)
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s...' % (url, name))
            r = os.system('curl -L %s -o %s' % (url, name))  # torch.hub.download_url_to_file(url, weights)
        finally:
            if not (os.path.exists(name) and os.path.getsize(name) > 1E6):  # check
                os.remove(name) if os.path.exists(name) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return
