# About:
Example csv files, which are used to specify which images belong to the two domains "Source1" and "Source2".
# Details:
The implementation assumes that the csv files are located in the folder "dataset_lowest_folder", which is specified 
via the config files and that all images referenced in the csv files are also located in the same folder or in its subfolders.
Hence, the path of the images, as defined in the column "img_path" in the csv file, is appended to "dataset_lowest_folder".
If the images are located directly "dataset_lowest_folder" the columns "img_path" and "file_name" should be identical.
The column "origin" refers to the name of the domain the image belongs to. 
Please keep the same order of occurrence of the domains for all csv files.
The column "issues" of the images can be used to filter out images for training, e.g. due to penmarks. 
An empty entry reflects no issues. All images should be of the same size and have equal height and width.
