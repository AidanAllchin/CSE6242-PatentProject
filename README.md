# CSE6242-PatentProject

GitHub repo for our project.

## Details

Data access link: `https://patentsview.org/download/data-download-tables`

## Installation and Setup

Follow these steps to set up the project:

1. Clone the repository:

```
git clone https://github.com/AidanAllchin/CSE6242-PatentProject.git
cd CSE6242-PatentProject
```

2. Create and activate a new virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install required packages:

```
pip install -r requirements.txt
```

## Usage

**FOR THE MOMENT, THE CORRECT USAGE TO DOWNLOAD AND CLEAN THE DATA IS TO MANUALLY RUN THE PYTHON NOTEBOOK.** This will change once I modify `main.py` to use the new data cleaning steps that Reid laid out. Ignore the following.

1. The script `python main.py` will walk through the steps to generate the SQLite database we'll be using for the remainder of the project. It's the only file that needs to be run manually after running `pip install -r requirements.txt`.

This script will:

- Create necessary directories
- Download the correct version of the dataset
- Split the large patent xml file into one file per patent
- Remove invalid patent files
- Parse relevant information and convert each patent to a **Patent** object
- Add details including: tags, coordinates for assignee/inventor, etc.
- Serialize all data into SQLite
