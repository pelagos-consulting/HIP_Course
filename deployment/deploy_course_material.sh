#!/bin/bash

cd ../course_material
chmod 755 ../deployment/convert_ipynb_to_html.py
../deployment/convert_ipynb_to_html.py
git add .
git commit -am "Latest"
git push
#cd ../
#zip -r acacia_training.zip acacia_training -x acacia_training/.git/\* -x *.ipynb_checkpoints\*
#rclone copy acacia_training.zip acacia-mine:acacia-training-2022/ --progress
