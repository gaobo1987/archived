import setuptools

description = 'This package provides the service that segments a given piece of text, ' \
              'typically short, such as a search query from user.'

setuptools.setup(
    name='online-segmentation',
    version='0.5.0',
    author='Bo Gao',
    author_email='bogao@huawei.com',
    description=description,
    long_description=description,
    long_description_content_type='text/markdown',
    url='https://git.huawei.com/BigData_Platform/BD_netherlands',
    packages=setuptools.find_packages(),
    install_requires=[
        'spacy',
        'torch',
        'transformers',
        'tensorboard'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
