# CSE6242-PatentProject

## Details

### Data Used (Automatic)

Note that **manual downloading of these files is not required**. Multiple files require specific file locations and naming conventions, so `__init__.py` (and by extension, `main.py`) will download, rename, and move these files automatically to ensure consistency.

|   Source   |           Table Name           | Link                                                                                        |
| :--------: | :----------------------------: | :------------------------------------------------------------------------------------------ |
| **USPTO**  |           `g_patent`           | https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip                     |
| **USPTO**  | `g_inventor_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_inventor_not_disambiguated.tsv.zip |
| **USPTO**  | `g_location_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_location_not_disambiguated.tsv.zip |
| **USPTO**  | `g_assignee_not_disambiguated` | https://s3.amazonaws.com/data.patentsview.org/download/g_assignee_not_disambiguated.tsv.zip |
| **USPTO**  |      `g_wipo_technology`       | https://s3.amazonaws.com/data.patentsview.org/download/g_wipo_technology.tsv.zip            |
|  **BEA**   | `CAINC1__ALL_AREAS_1969_2023`  | https://apps.bea.gov/regional/zip/CAINC1.zip                                                |
|  **BEA**   | `CAGDP1__ALL_AREAS_2001_2022`  | https://apps.bea.gov/regional/zip/CAGDP1.zip                                                |
|  **BEA**   | `CAINC4__ALL_AREAS_1969_2023`  | https://apps.bea.gov/regional/zip/CAINC4.zip                                                |
|  **BEA**   | `CAINC30__ALL_AREAS_1969_2023` | https://apps.bea.gov/regional/zip/CAINC30.zip                                               |
| **Census** |  `county_boundaries.geojson`   | https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/us-county-boundaries/exports/geojson?lang=en&timezone=America%2FNew_York |

### Data Used (Manual)

THe following tables should be downloaded manually and placed in their respective directories unless you wish to run the full pipeline (including running a 32B-parameter LLM locally), which can take >24 hours or simply be impossible depending on your device.

You can generate the third file using the first two files, or can download all three and jump straight to generating the BEA features and training the model.

| **Link** | **File Location** | **Size (MB)** | **Time to Generate (hr)** | **Requires LLM** |
| :------- | :--------------- | :-----------: | :-----------------------: | :---------: |
| https://drive.google.com/file/d/178rhI4UhdwRtPUNkZPrOhQRl5pVFtL1f/view?usp=drive_link | `CSE6242-PatentProject/data/geolocation/` | 1.4 | 6  | n |
| https://drive.google.com/file/d/1NPTZyfcBFptAvhmcSngn_Fo8VhfT9Gi-/view?usp=drive_link | `CSE6242-PatentProject/data/geolocation/` | 0.8 | 10 | y |
| https://drive.google.com/file/d/18AX7vQeApAuCK2KgTLUACeJf36D2Band/view?usp=drive_link | `CSE6242-PatentProject/data/` | 913.6 | 0.4 | n |


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
- **Menu Item 2:** Load BEA data
- **Menu Item 3:** (Unused) Load census data
- **Menu Item 4:** (Unused) Load Fed data
- **Menu Item 5:** Group patents and create model metrics organized by time window and county
- **Menu Item 5:** Create predictors and `innovation_score` for the Innovation Hub Predictor
- **Menu Item 6:** Train IHPM and predict next period innovation score for all counties for overlay
- **Menu Item 6:** Add latitude and longitude for the predicted values back to the data for the overlay


## Known Issues

The BEA tables downloaded in `__init__.py` have updated since starting the project. I've fixed the links for 3 of the 4, but `CAGDP1__ALL_AREAS_2001_2022` contains the GDP information per-county (one of our most important metrics) and still hasn't been updated to include 2023 as of November 15th. If the unzipping step fails for that file, change the "2022" in line 188 to "2023".


