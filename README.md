# Caltech Pedestrian Dataset Extractor

Extract images and annotation files from the Caltech Pedestrian Dataset using python.

This script extracts and converts data from `.seq` and `.vbb` files  to `.jpg` images and `.json` files. Individual images + annotation files are extracted and stored per sequence and per video for easier access.


# Requirements

- python>=2.7 or python>=3.3
- numpy
- scipy
- json


# Usage

In order to start using this script you must first untar all files before running the script.

Then, to extract files just run the following comand in a terminal:

```bash
python converter.py -data_path <path_to_data>
```

This will create a dir named `extracted_data/` and store all extracted data in there. If you wish to specify a different directory to store your data you can set the `-save_path` input argument to a different path.

```bash
python converter.py -data_path <path_to_data> -save_path <path_to_extract>
```

> Note: A valid `-data_path` is required and must be provided in order for the script to work.


# License

Licensed under the [MIT](LICENSE) license.


# Acknowledgements

This code is based on @hizhangp [caltech converter](https://github.com/hizhangp/caltech-pedestrian-converter) code.