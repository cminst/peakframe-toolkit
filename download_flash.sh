#!/bin/bash

echo "Downloading FLASH dataset from HuggingFace..."
hf download qingy2024/FLASH-Dataset flash_videos.tar.gz --repo-type=dataset --local-dir .
tar -xzvf flash_videos.tar.gz
rm flash_videos.tar.gz
mv downloaded_clips/ data/
echo "Finished downloading FLASH dataset!"
