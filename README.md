# CSE6242-PatentProject

GitHub repo for our project.

## Details

### Data Used

Note that **manual downloading of these files is not required**. Multiple files require specific file locations and naming conventions, so `__init__.py` (and by extension, `main.py`) will download, rename, and move these files automatically to ensure consistency.

|   Source   |           Table Name           | Link                                                                                        |
| :--------: | :----------------------------: | :------------------------------------------------------------------------------------------ |
| **USPTO**  |           `g_patent`           | https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip                     |
| **USPTO**  | `g_inventor_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_inventor_not_disambiguated.tsv.zip |
| **USPTO**  | `g_location_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_location_not_disambiguated.tsv.zip |
| **USPTO**  | `g_assignee_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_assignee_not_disambiguated.tsv.zip |
| **USPTO**  |      `g_wipo_technology`       | https://s3.amazonaws.com/data.patentsview.org/download/g_wipo_technology.tsv.zip            |
|  **BEA**   | `CAINC1__ALL_AREAS_1969_2022`  | https://apps.bea.gov/regional/zip/CAINC1.zip                                                |
|  **BEA**   | `CAGDP1__ALL_AREAS_2001_2022`  | https://apps.bea.gov/regional/zip/CAGDP1.zip                                                |
|  **BEA**   | `CAINC4__ALL_AREAS_1969_2022`  | https://apps.bea.gov/regional/zip/CAINC4.zip                                                |
|  **BEA**   | `CAINC30__ALL_AREAS_1969_2022` | https://apps.bea.gov/regional/zip/CAINC30.zip                                               |
| **Census** |     `fips_to_county_name`      | https://raw.githubusercontent.com/ChuckConnell/articles/refs/heads/master/fips2county.tsv   |
| **Census** |  `county_boundaries.geojson`   | https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/us-county-boundaries/exports/geojson?lang=en&timezone=America%2FNew_York |

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

OR use `conda`, `mamba`, or another environment manager. 

3. Install required packages:

```
pip install -r requirements.txt
```

## Usage

1. Requires downloading of `city_coordinates.tsv` and `location_corrections.tsv` from the SharePoint in the `data` folder. 

- Pull the GitHub repo
- Run `python main.py`
- Before selecting a menu option, place these two files in the newly created `/data/geolocation` folder
- Proceed with step 2&3

2. To avoid having to run the patent cleaning pipeline yourselves, I've also uploaded the final `patents.tsv`. By downloading this file (also from the SharePoint) and placing it in `/data`, you can skip to menu item #2 in `main.py`.

3. The script `python main.py` will walk through the steps to generate the `.tsv` files we'll be using for the remainder of the project. It's the only file that needs to be run. The menu items are designed to be run sequentially.

This script will:

- Install required packages if needed
- Create necessary directories
- Download all required tables
- **Menu Item 1:** Merge all tables
- **Menu Item 1:** Perform all data cleaning steps
- **Menu Item 1:** Add latitude and longitude for inventor and assignee to each patent
- **Menu Item 1:** Add inventor origination county information to every patent based on coordinates
- **Menu Item 1:** Generate a `patents.tsv` file with all US-based patents since 2001 (this time constraint is due to GDP data being unavailable per-county prior to this)
- **Menu Item 2:** WIP: Load BEA data
- **Menu Item 3:** WIP: Load census data
- **Menu Item 4:** WIP: Load Fed data
- **Menu Item 5:** WIP: Group patents and create model metrics organized by time window and county
- **Menu Item 5:** WIP: Create predictors and `innovation_score` for the Innovation Hub Predictor
- WIP: Use generated model to predict next period innovation score for any county
