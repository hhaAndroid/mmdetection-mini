import argparse
from cvcore import DictAction, Config, convert_image_to_rgb, Logger
from detecter.dataset import build_dataset
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def _recursion_dict_to_list(results, output_keys):
    output_keys.append(list(results.keys()))
    for key, value in results.items():
        if isinstance(value, dict):
            _recursion_dict_to_list(value, output_keys)
    return output_keys


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    Logger.init()

    dataset_cfg = eval(f'cfg.data.{args.mode}')
    dataset_class = build_dataset(dataset_cfg, default_args=dict(_no_instantiate_=True))

    def prepare_img(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        self.pre_pipeline(results)

        for i, t in enumerate(self.pipeline.transforms):
            print('===================')
            output_keys_pre = []
            _recursion_dict_to_list(results, output_keys_pre)
            print('pre', output_keys_pre)
            results = t(results)

            output_keys_post = []
            _recursion_dict_to_list(results, output_keys_post)
            print('post', output_keys_post)

            # 新增 key


        return results

    dataset_class.prepare_img = prepare_img
    dataset_cfg.pop('type')
    dataset = dataset_class(**dataset_cfg)

    for data in dataset:
        print(data)


if __name__ == '__main__':
    main()
