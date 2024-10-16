# CSE6242-PatentProject

GitHub repo for our project.

## Details

Current data release date: **2024-09-26**

Data access link: `https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/`

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

3. Run the initialization script:

```
python __init__.py
```

This script will:

- Install required packages from `requirements.txt`
- Create necessary directories
- Download the correct version of the dataset

## Usage

1. Eventually `main.py` will hold all this junk but for now run the following to split the data into individual xml files:

```
python src/data_cleaning/xml_splitter.py
```

It should split the data file into more manageable file sizes, each one corresponding to the ID of a patent in the larger data.
