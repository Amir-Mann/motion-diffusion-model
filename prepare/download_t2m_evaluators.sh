echo -e "Downloading T2M evaluators"
/home/amir.mann/.local/bin/gdown --fuzzy https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view
/home/amir.mann/.local/bin/gdown --fuzzy https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view
rm -rf t2m
rm -rf kit

unzip t2m.zip
unzip kit.zip
echo -e "Cleaning\n"
rm t2m.zip
rm kit.zip

echo -e "Downloading done!"
