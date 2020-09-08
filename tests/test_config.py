# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import tempfile

from mmdet.cv_core import Config, dump, load

data_path = osp.join(osp.dirname(osp.dirname(__file__)), 'tests/data')


def test_pickle_support():
    cfg_file = osp.join(data_path, 'config/n.py')
    cfg = Config.fromfile(cfg_file)

    with tempfile.TemporaryDirectory() as temp_config_dir:
        pkl_cfg_filename = osp.join(temp_config_dir, '_pickle.pkl')
        dump(cfg, pkl_cfg_filename)
        pkl_cfg = load(pkl_cfg_filename)

    assert pkl_cfg._cfg_dict == cfg._cfg_dict


if __name__ == '__main__':
    test_pickle_support()
