# CSE6242-PatentProject

GitHub repo for our project.

## Details

### Data Used

|  Source   |           Table Name           |                                            Link                                             |
| :-------: | :----------------------------: | :-----------------------------------------------------------------------------------------: |
| **USPTO** |           `g_patent`           |           https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip           |
| **USPTO** | `g_inventor_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_inventor_not_disambiguated.tsv.zip |
| **USPTO** | `g_location_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_location_not_disambiguated.tsv.zip |
| **USPTO** | `g_assignee_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_assignee_not_disambiguated.tsv.zip |
| **USPTO** |      `g_wipo_technology`       |      https://s3.amazonaws.com/data.patentsview.org/download/g_wipo_technology.tsv.zip       |
|  **BEA**  | `CAINC1__ALL_AREAS_1969_2022`  |                        https://apps.bea.gov/regional/zip/CAINC1.zip                         |
|  **BEA**  | `CAGDP1__ALL_AREAS_2001_2022`  |                        https://apps.bea.gov/regional/zip/CAGDP1.zip                         |
|  **BEA**  | `CAINC4__ALL_AREAS_1969_2022`  |                        https://apps.bea.gov/regional/zip/CAINC4.zip                         |
|  **BEA**  | `CAINC30__ALL_AREAS_1969_2022` |                        https://apps.bea.gov/regional/zip/CAINC30.zip                        |

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

1. The script `python main.py` will walk through the steps to generate the `.tsv` files we'll be using for the remainder of the project. It's the only file that needs to be run. The menu items are designed to be run sequentially.

This script will:

- Install required packages if needed
- Create necessary directories
- Download all required tables
- Merge all tables
- Perform all data cleaning steps
- Generate a `processed_patents.tsv` file with all US-based patents since 2001 (this time constraint may no longer be necessary)
- WIP: Add county information to every patent
- WIP: Group patents and create model metrics organized by time window and county
- WIP: Load census data
- WIP: Load BEA data
- WIP: Load Fed data
- WIP: Create predictors and `innovation_score` for the Innovation Hub Predictor
- WIP: Use generated model to predict next period innovation score for any county
