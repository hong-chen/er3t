# pack data
# tar -czf er3t-data.tar.gz er3t/data

# download data from google drive
# ====================================================================================================
if ! command -v gdown &> /dev/null
then
    echo "[Error]: <gdown> could not be found, please install <gdown> first, abort."
    exit
fi

er3t_data_google_id="1KKpLR7IyqJ4gS6xCxc7f1hwUfUMJksVL"

er3t_data_local_filename="er3t-data.tar.gz"

echo "1. Install Data ########################################"
echo
sleep 1

echo "<1.1> Downloading required data ... (this will take minutes to hours depending on your internet speed)"
echo "Start =================================================="
echo
echo "command: gdown $er3t_data_google_id --output $er3t_data_local_filename"
echo
gdown $er3t_data_google_id --output $er3t_data_local_filename
echo "Complete ==============================================="

echo
sleep 2

# check if file is successfully downloaded
if  [ ! -f "$er3t_data_local_filename" ]
then
    echo "[Error]: cannot find <$er3t_data_local_filename>, abort."
    exit
fi

echo "<1.2> Untaring the downloaded data ... (this will take a few minutes)"
echo "Start =================================================="
echo
echo "command: tar -xzf $er3t_data_local_filename"
echo
tar -xzf $er3t_data_local_filename
echo "Complete ==============================================="

echo
sleep 2

echo "<1.3> Cleaning up ..."
echo "Start =================================================="
echo
echo "command: rm -rf $er3t_data_local_filename"
echo
rm -rf $er3t_data_local_filename
echo "Complete ==============================================="

# ====================================================================================================

echo

if [ -f setup.py ]
then
    python setup.py develop
fi
