import glob
import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDAExtension, CppExtension, BuildExtension)


# 获取当前库版本号
def get_version():
    version_file = 'mmdet/__about__.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_extensions():
    extensions = []
    ext_name = 'mmdet.cv_core._ext'
    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available():
        print(f'Compiling {ext_name} with CUDA')
        define_macros += [('MMCV_WITH_CUDA', None)]
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        op_files = glob.glob('./mmdet/cv_core/ops/csrc/pytorch/*')
        extension = CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./mmdet/cv_core/ops/csrc/pytorch/*.cpp')
        extension = CppExtension

    include_path = os.path.abspath('./mmdet/cv_core/ops/csrc')
    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)
    return extensions


setup(
    name='mmdetection-mini',
    version=get_version(),
    description='OpenMMLab Computer Vision Foundation',
    keywords='computer vision',
    packages=find_packages(exclude=('configs', 'tools', 'demo', 'tests', 'docs')),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
    ],
    url='https://github.com/hhaAndroid/mmdetection-mini',
    author='HHA Authors',
    author_email='1286304229@qq.com',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False)
