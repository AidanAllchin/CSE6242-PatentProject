#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

This script is used to parse the individual patent XML files to create Patent
objects. 
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
from typing import List, Dict, Tuple
import re
import time
from tqdm import tqdm
import json
from lxml import etree
from src.objects.patent import Patent
from src.other.helpers import get_patent_id_from_filename, log, local_filename


###############################################################################
#                               CONFIGURATION                                 #
###############################################################################


# Directories
PATENTS_DIRECTORY = os.path.join(project_root, 'data', 'patents')
CONFIG_PATH = os.path.join(project_root, 'config', 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

skip_count = 0


###############################################################################
#                             INDIVIDUAL PARSER                               #
###############################################################################

""" Generalized XML structure and tags:
Patent Name (Invention Title):                   DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <invention-title>


Assignee (Company):                              DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <us-parties>
         <us-applicants>
             <us-applicant>
                 <addressbook>
                     <orgname>
                     <address>
                         <city>
                         <state>
                         <country>

Inventors:                                       DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <us-parties>
         <inventors>
             <inventor> (often multiple)
                 <addressbook>
                     <last-name>
                     <first-name>
                     <address>
                         <city>
                         <state>
                         <country>

Dates:                                           DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <application-reference>
         <document-id>
             <date>
 <publication-reference>
     <document-id>
         <date>
 date-produced (attribute of <us-patent-application>)
 date-publ (attribute of <us-patent-application>)

Classifications/Industries:                      DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <classifications-ipcr>     (International Patent Classification)
     (these are used internationally and are less specific)
         <classification-ipcr>
             <section>
             <class>
             <subclass>
             <main-group>
             <subgroup>
     <classifications-cpc>      (Cooperative Patent Classification)
     (used by USPTO and EPO, more specific, updated more regularly)
         <main-cpc>
             <classification-cpc>
                 <section>
                 <class>
                 <subclass>
                 <main-group>
                 <subgroup>

Application Number:                              DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <application-reference>
         <document-id>
             <doc-number>

Document Kind:                                   DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <publication-reference>
         <document-id>
             <kind>

Application Type:                                DONE
<us-patent-application>
 <us-bibliographic-data-application>
     <application-reference appl-type>


Other garbage:

Abstract:                                        DONE
<us-patent-application>
  <abstract>

Claims:                                          DONE
<us-patent-application>
  <claims>
      <claim>

Description:                                     DONE
<us-patent-application>
  <description>
"""

def extract_patent_name(file_loc: str) -> str:
    """
    Extracts the name of the patent from the XML file.

    Args:
        file_loc (str): The location of the XML file to extract the patent name from.

    Returns:
        str: The name of the patent.
    """
    patent_id = get_patent_id_from_filename(file_loc)

    try:
        tree = etree.parse(file_loc)
        invention_title = tree.xpath(f"//invention-title")
        if invention_title:
            # Shouldn't be more than one but it's returned as a list
            if len(invention_title) > 1:
                log(f"Multiple invention titles found for patent {patent_id}. No action required - taking first instance.", level="WARNING")
            return invention_title[0].text.strip()
        else:
            pass
            #log(f"No invention title found for patent {patent_id}.", level="ERROR")
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}.", level="ERROR")
    return None

def extract_assignee_info(file_loc: str) -> List[dict]:
    """
    Extracts assignee information from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract assignee information from.

    Returns:
        list: A list of dictionaries containing assignee information.
    """
    patent_id = get_patent_id_from_filename(file_loc)
    assignee_info = []

    def parse_assignee(assignee: etree.Element) -> dict:
        """
        Cute lil helper function to parse assignee info.

        Args:
            assignee (etree.Element): The assignee element to parse.

        Returns:
            dict: The parsed assignee information.
        """
        return {
            # Pretty sure these are the only fields we care about
            "name": assignee.xpath('.//orgname/text()')[0].strip() if assignee.xpath('.//orgname') else None,
            "city": assignee.xpath('.//city/text()')[0].strip() if assignee.xpath('.//city') else None,
            "state": assignee.xpath('.//state/text()')[0].strip() if assignee.xpath('.//state') else None,
            "country": assignee.xpath('.//country/text()')[0].strip() if assignee.xpath('.//country') else None
        }

    try:
        tree = etree.parse(file_loc)
        assignees = tree.xpath(f"//us-applicant")
        if not assignees:
            log(f"No assignees found for patent {patent_id}.", level="WARNING")
            return None
        
        # Loop through all assignees and parse their info
        for assignee in assignees:
            assignee_info.append(parse_assignee(assignee))
        
        return assignee_info
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}.", level="ERROR")
    return None

def extract_inventors(file_loc: str) -> List[dict]:
    """
    Extracts inventor information from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract inventor information from.

    Returns:
        list: A list of dictionaries containing inventor information.
    """
    patent_id = get_patent_id_from_filename(file_loc)
    inventor_info = []

    def parse_inventor(inventor: etree.Element) -> dict:
        """
        Another helper function to parse inventor info.

        Args:
            inventor (etree.Element): The inventor element to parse.

        Returns:
            dict: The parsed inventor information.
        """
        return {
            "last_name": inventor.xpath('.//last-name/text()')[0].strip() if inventor.xpath('.//last-name') else None,
            "first_name": inventor.xpath('.//first-name/text()')[0].strip() if inventor.xpath('.//first-name') else None,
            "city": inventor.xpath('.//city/text()')[0].strip() if inventor.xpath('.//city') else None,
            "state": inventor.xpath('.//state/text()')[0].strip() if inventor.xpath('.//state') else None,
            "country": inventor.xpath('.//country/text()')[0].strip() if inventor.xpath('.//country') else None
        }

    try:
        tree = etree.parse(file_loc)
        inventors = tree.xpath(f"//inventors/inventor")
        if not inventors:
            log(f"No inventors found for patent {patent_id}.", level="WARNING")
            return None
        
        # Loop through all inventors and parse their info
        for inventor in inventors:
            inventor_info.append(parse_inventor(inventor))
        
        return inventor_info
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}.", level="ERROR")
    return None

def extract_dates(file_loc: str) -> dict:
    """
    Extracts relevant dates from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract dates from.

    Returns:
        dict: A dictionary containing the extracted dates.
    """
    patent_id = get_patent_id_from_filename(file_loc)
    dates = {}

    def convert_date(d: str) -> str:
        """
        Converts from the format YYYYMMDD to YYYY-MM-DD.

        Args:
            d (str): The date to convert.

        Returns:
            str: The converted date.
        """
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    
    try:
        tree = etree.parse(file_loc)
        
        # Extract application date
        application_date = tree.xpath("//application-reference/document-id/date/text()")
        if application_date:
            dates['application_date'] = convert_date(application_date[0].strip())

        # Extract publication date
        publication_date = tree.xpath("//publication-reference/document-id/date/text()")
        if publication_date:
            dates['publication_date'] = convert_date(publication_date[0].strip())

        # Extract date-produced and date-publ from root element attributes
        root = tree.getroot()
        dates['date_produced'] = convert_date(root.get('date-produced')) if root.get('date-produced') else None
        dates['date_publ']     = convert_date(root.get('date-publ')) if root.get('date-publ') else None

        if not dates:
            log(f"No dates found for patent {patent_id}.", level="WARNING")
            return None

        return dates
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}.", level="ERROR")
    return None

def extract_classifications(file_loc: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Extracts IPC and CPC classifications from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract classifications from.

    Returns:
        dict: A dictionary containing two lists of classifications (IPC and CPC).
    """
    patent_id = get_patent_id_from_filename(file_loc)
    classifications = {"ipc": [], "cpc": []}

    def parse_ipc(ipc_elem: etree.Element) -> Dict[str, str]:
        """Helper function to parse IPC classification."""
        return {
            "section": ipc_elem.xpath('./section/text()')[0] if ipc_elem.xpath('./section') else None,
            "class": ipc_elem.xpath('./class/text()')[0] if ipc_elem.xpath('./class') else None,
            "subclass": ipc_elem.xpath('./subclass/text()')[0] if ipc_elem.xpath('./subclass') else None,
            "main_group": ipc_elem.xpath('./main-group/text()')[0] if ipc_elem.xpath('./main-group') else None,
            "subgroup": ipc_elem.xpath('./subgroup/text()')[0] if ipc_elem.xpath('./subgroup') else None,
        }

    def parse_cpc(cpc_elem: etree.Element) -> Dict[str, str]:
        """Helper function to parse CPC classification."""
        return {
            "section": cpc_elem.xpath('./section/text()')[0] if cpc_elem.xpath('./section') else None,
            "class": cpc_elem.xpath('./class/text()')[0] if cpc_elem.xpath('./class') else None,
            "subclass": cpc_elem.xpath('./subclass/text()')[0] if cpc_elem.xpath('./subclass') else None,
            "main_group": cpc_elem.xpath('./main-group/text()')[0] if cpc_elem.xpath('./main-group') else None,
            "subgroup": cpc_elem.xpath('./subgroup/text()')[0] if cpc_elem.xpath('./subgroup') else None,
        }

    try:
        tree = etree.parse(file_loc)
        
        # Extract IPC classifications
        ipc_elems = tree.xpath("//classifications-ipcr/classification-ipcr")
        for ipc in ipc_elems:
            classifications["ipc"].append(parse_ipc(ipc))
        
        # Extract CPC classifications
        cpc_elems = tree.xpath("//classifications-cpc//classification-cpc")
        for cpc in cpc_elems:
            classifications["cpc"].append(parse_cpc(cpc))

        if not classifications["ipc"] and not classifications["cpc"]:
            log(f"No classifications found for patent {patent_id}.", level="WARNING")
            return None

        return classifications
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}.", level="ERROR")
    return None

def extract_application_info(file_loc: str) -> Tuple[str, str, str]:
    """
    Extracts application number, document kind, and application type from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract application info from.

    Returns:
        tuple: A tuple containing the application number, document kind, and application type.
    """
    patent_id = get_patent_id_from_filename(file_loc)
    application_number, document_kind, application_type = None, None, None

    try:
        tree = etree.parse(file_loc)
        
        # Extract application number
        app_number = tree.xpath("//application-reference/document-id/doc-number/text()")
        if app_number:
            application_number = app_number[0].strip()
        else:
            log("Application number not found", level="WARNING")

        # Extract document kind
        doc_kind = tree.xpath("//publication-reference/document-id/kind/text()")
        if doc_kind:
            document_kind = doc_kind[0].strip()
        else:
            log("Document kind not found", level="WARNING")

        # Extract application type
        app_type = tree.xpath("//application-reference/@appl-type")
        if app_type:
            application_type = app_type[0].strip()
        else:
            log("Application type not found", level="WARNING")

        if not application_number and not document_kind and not application_type:
            log(f"No application info found for patent {patent_id}", level="WARNING")
            return None, None, None

        return application_number, document_kind, application_type
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}", level="ERROR")
    return None, None, None

def extract_abstract(file_loc: str) -> str:
    """
    Extracts the abstract from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract the abstract from.

    Returns:
        str: The abstract text, or None if not found.
    """
    patent_id = get_patent_id_from_filename(file_loc)
    
    try:
        tree = etree.parse(file_loc)
        abstract_elem = tree.xpath("//abstract")
        if abstract_elem:
            # Extract all text from the abstract, including nested elements
            abstract_text = ' '.join(abstract_elem[0].xpath(".//text()")).strip()
            return abstract_text
        else:
            log(f"No abstract found for patent {patent_id}", level="WARNING")
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}", level="ERROR")
    return None

def extract_claims(file_loc: str) -> Dict[str, str]:
    """
    Extracts the claims from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract the claims from.

    Returns:
        Dict[str, str]: A dictionary containing the extracted claims.
    """
    patent_id = get_patent_id_from_filename(file_loc)
    
    try:
        tree = etree.parse(file_loc)
        claims = tree.xpath("//claims/claim")
        if claims:
            extracted_claims = []
            for claim in claims:
                claim_num = claim.get('num', 'Unknown')
                claim_text = ' '.join(claim.xpath(".//text()")).strip()
                extracted_claims.append({'number': claim_num, 'text': claim_text})
            return extracted_claims
        else:
            log(f"No claims found for patent {patent_id}", level="WARNING")
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}", level="ERROR")
    return None

def extract_description(file_loc: str) -> str:
    """
    Extracts the description from a patent XML file.

    Args:
        file_loc (str): The location of the XML file to extract the description from.

    Returns:
        str: The description text, or None if not found.
    """
    patent_id = get_patent_id_from_filename(file_loc)
    
    try:
        tree = etree.parse(file_loc)
        description_elem = tree.xpath("//description")
        if description_elem:
            # Extract all text from the description, including nested elements
            description_text = ' '.join(description_elem[0].xpath(".//text()")).strip()
            return description_text
        else:
            log(f"No description found for patent {patent_id}", level="WARNING")
    except etree.XMLSyntaxError:
        log(f"Error parsing XML file for patent {patent_id}", level="ERROR")
    return None


def parse_one(file_loc: str):
    """
    Parses a single patent file to extract the relevant information.

    Args:
        file_loc (str): The location of the XML file to parse.
    """
    if not os.path.exists(file_loc):
        log(f"File {file_loc} not found.", level="ERROR")
        return
    
    # Give me everything
    patent_id       = get_patent_id_from_filename(file_loc)
    p_name          = extract_patent_name(file_loc)

    # Don't bother parsing if there's no name
    if not p_name:
        global skip_count
        #log(f"Patent {patent_id} has no name. Skipping.", level="WARNING")
        skip_count += 1
        return None

    asignee_info    = extract_assignee_info(file_loc)
    inventor_info   = extract_inventors(file_loc)
    dates_info      = extract_dates(file_loc)
    classifications = extract_classifications(file_loc)
    num, kind, t    = extract_application_info(file_loc)
    abstract        = extract_abstract(file_loc)
    claims          = extract_claims(file_loc)
    description     = extract_description(file_loc)

    # TEMP: Print all the data
    # log(f"Parsed patent {patent_id} with name: {p_name}", color_full=True)
    # log(f"Assignee info for patent {patent_id}: {asignee_info}")
    # log(f"Inventor info for patent {patent_id}: {inventor_info}")
    # log(f"Dates info for patent {patent_id}: {dates_info}")
    # log(f"Classifications for patent {patent_id}: {classifications}")
    # log(f"Application info for patent {patent_id}: {num}, {kind}, {t}")
    # log(f"Abstract for patent {patent_id}: {abstract}")
    # log(f"Patent {patent_id} has {len(claims)} claims. First: {claims[0] if claims else None}")
    # log(f"Description for patent {patent_id} is {len(description)} characters long.")

    # Convert to Patent object
    p = Patent(
        patent_id=int(patent_id),
        patent_name=p_name,
        assignee_info=asignee_info,
        inventor_info=inventor_info,
        dates_info=dates_info,
        classifications=classifications,
        application_number=int(num),
        document_kind=kind,
        application_type=t,
        abstract=abstract,
        claims=claims,
        description=description
    )

    return p



###############################################################################
#                                   MAIN                                      #
###############################################################################

def collect_patent_objects() -> List[Patent]:
    if not os.path.exists(PATENTS_DIRECTORY):
        log(f"Directory {PATENTS_DIRECTORY} not found.", level="ERROR")
        sys.exit(1)
    
    # Get all patent files in the directory
    patent_files = os.listdir(PATENTS_DIRECTORY)
    patent_files = [f for f in patent_files if f.endswith('.xml')]
    #patent_ids   = [re.search(r'\d+', f).group() for f in patent_files]

    log(f"Found {len(patent_files)} patent files in {local_filename(PATENTS_DIRECTORY)}.\n", color=Fore.CYAN, color_full=True)
    
    # Parse a single patent file
    #print(parse_one(os.path.join(PATENTS_DIRECTORY, "patent_20240317807.xml")))#patent_files[2])))
    #sys.exit(0)

    # Parse each patent file
    start = time.time()
    st    = time.time()
    times = []
    patent_objects = []
    for i in tqdm(patent_files, desc="Creating Patent Objects"):
        full_name = os.path.join(PATENTS_DIRECTORY, i)
        patent_objects.append(parse_one(full_name))
        times.append(time.time() - st)
        st = time.time()
    log(f"Average time to parse a patent file: {sum(times) / len(times):.2f}s.")

    if skip_count:
        log(f"\nSkipped {skip_count} patents due to missing names.", level="WARNING")
    log(f"Generated {len(patent_files) - skip_count} patent files in {time.time() - start:.2f}s.", color=Fore.LIGHTBLUE_EX, color_full=True)

    # print(patent_objects[0])

    return [p for p in patent_objects if p is not None]


if __name__ == '__main__':
    patents = collect_patent_objects()

    # Count how many have inventors in the US
    us_inventors = 0
    for p in patents:
        for i in p.inventor_info:
            if i['country'] == 'US':
                us_inventors += 1
                break

    log(f"\n{us_inventors} / {len(patents)} patents have inventors in the US.", color=Fore.LIGHTGREEN_EX, color_full=True)

