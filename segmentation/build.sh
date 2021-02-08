#!/bin/bash

rm -r release_wheels
mkdir release_wheels
for lang_code in nl pl pt el cs
do
    rm -rf build/
    rm -rf dist/
    rm -rf online_segmentation.egg-info/
    rm MANIFEST.in
    python create_language_manifest.py $lang_code
    python setup.py sdist bdist_wheel
    mv dist/online_segmentation-$1-py3-none-any.whl release_wheels/online_segmentation-$1-${lang_code}-py3-none-any.whl
done

rm -rf build/
rm -rf dist/
rm -rf online_segmentation.egg-info/
