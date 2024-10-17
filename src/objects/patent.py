#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Tue Oct 17 2024
Author: Aidan Allchin

This holds the object class for a Patent.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
from typing import Dict
from datetime import datetime
import sqlite3

class Patent:
    def __init__(self, patent_id: int, patent_name: str, assignee_info: list, inventor_info: list, dates_info: Dict, classifications: Dict, application_number: int, document_kind: str, application_type: str, abstract: str, claims: list, description: str):
        self.patent_id = patent_id
        self.patent_name = patent_name
        self.assignee_info = assignee_info
        self.inventor_info = inventor_info
        self.dates_info = dates_info
        self.classifications = classifications
        self.application_number = application_number
        self.document_kind = document_kind
        self.application_type = application_type
        self.abstract = abstract
        self.claims = claims
        self.description = description
        
        self.__validate__()

    def __validate__(self):
        """
        Validates the patent object to ensure all required fields are present
        and all types are correct and consistent.
        """
        assert isinstance(self.patent_id, int), f"Patent ID must be an integer, not {type(self.patent_id)}"
        assert len(str(self.patent_id)) == 11, f"Patent ID must be 11 characters long, not {len(str(self.patent_id))}" # TODO: Double check this
        assert isinstance(self.patent_name, str), f"Patent Name must be a string, not {type(self.patent_name)}"
        assert isinstance(self.assignee_info, list), f"Assignee Info must be a list, not {type(self.assignee_info)}"
        assert isinstance(self.inventor_info, list), f"Inventor Info must be a list, not {type(self.inventor_info)}"
        assert isinstance(self.dates_info, dict), f"Dates Info must be a dictionary, not {type(self.dates_info)}"
        assert isinstance(self.classifications, dict), f"Classifications must be a dictionary, not {type(self.classifications)}"
        assert isinstance(self.application_number, int), f"Application Number must be an integer, not {type(self.application_number)}"
        assert isinstance(self.document_kind, str), f"Document Kind must be a string, not {type(self.document_kind)}"
        assert isinstance(self.application_type, str), f"Application Type must be a string, not {type(self.application_type)}"
        
        if self.abstract is not None:
            assert isinstance(self.abstract, str), f"Abstract must be a string, not {type(self.abstract)}"
        if self.claims is not None:
            assert isinstance(self.claims, list), f"Claims must be a list, not {type(self.claims)}"
        if self.description is not None:
            assert isinstance(self.description, str), f"Description must be a string, not {type(self.description)}"
        
        for assignee in self.assignee_info:
            assert isinstance(assignee, dict), f"Assignee must be a dictionary, not {type(assignee)}"
            assert 'name' in assignee, "Assignee name missing"
            assert 'city' in assignee, "Assignee city missing"
            assert 'state' in assignee, "Assignee state missing"
            assert 'country' in assignee, "Assignee country missing"
            if assignee['country'] is not None:
                assert len(assignee['country']) == 2, f"Assignee country must be a 2-letter abbreviation, not {assignee['country']}"
            assert len(assignee) == 4, f"Assignee must have 4 fields, not {len(assignee)}"

        for inventor in self.inventor_info:
            assert isinstance(inventor, dict), f"Inventor must be a dictionary, not {type(inventor)}"
            assert 'last_name' in inventor, "Inventor last name missing"
            assert 'first_name' in inventor, "Inventor first name missing"
            assert 'city' in inventor, "Inventor city missing"
            assert 'state' in inventor, "Inventor state missing"
            assert 'country' in inventor, "Inventor country missing"
            if inventor['country'] is not None:
                assert len(inventor['country']) == 2, f"Inventor country must be a 2-letter abbreviation, not {inventor['country']}"
            assert len(inventor) == 5, f"Inventor must have 5 fields, not {len(inventor)}"

        for d_name, d_val in self.dates_info.items():
            assert isinstance(d_name, str), f"Date name must be a string, not {type(d_name)}"
            if isinstance(d_val, str):
                try:
                    datetime.strptime(d_val, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Date value must be in the format 'YYYY-MM-DD', not {d_val}")
            else:
                assert isinstance(d_val, datetime), f"Date value must be a datetime object, not {type(d_val)}"
        
        if self.classifications is not None:
            assert 'ipc' in self.classifications, "IPC classification missing"
            assert 'cpc' in self.classifications, "CPC classification missing"
        
        for dic in self.classifications['ipc']:
            assert isinstance(dic, dict), f"IPC classification must be a dictionary, not {type(dic)}"
            assert 'section' in dic, "IPC section missing"
            assert 'class' in dic, "IPC class missing"
            assert 'subclass' in dic, "IPC subclass missing"
            assert 'main_group' in dic, "IPC main group missing"
            assert 'subgroup' in dic, "IPC subgroup missing"
        
        for dic in self.classifications['cpc']:
            assert isinstance(dic, dict), f"CPC classification must be a dictionary, not {type(dic)}"
            assert 'section' in dic, "CPC section missing"
            assert 'class' in dic, "CPC class missing"
            assert 'subclass' in dic, "CPC subclass missing"
            assert 'main_group' in dic, "CPC main group missing"
            assert 'subgroup' in dic, "CPC subgroup missing"

        if self.claims is not None:
            for claim in self.claims:
                assert isinstance(claim, dict), f"Claim must be a dictionary, not {type(claim)}"
                assert 'number' in claim, "Claim number missing"
                assert 'text' in claim, "Claim text missing"
                assert len(claim) == 2, f"Each claim must have 2 fields, not {len(claim)}"
        self.ipc = self.classifications['ipc']
        self.cpc = self.classifications['cpc']
    
    def __str__(self):
        s = f"\n{Style.BRIGHT}{Fore.GREEN}Patent {self.patent_id}: {self.patent_name}{Style.RESET_ALL}"
        s += f"\n{Style.BRIGHT}Assignee Info:{Style.RESET_ALL}"
        for assignee in self.assignee_info:
            s += f"\n\t- {assignee['name']} ({assignee['country']}): {assignee['city']}, {assignee['state'] if assignee['state'] else 'N/A'}"

        s += f"\n\n{Style.BRIGHT}Inventor Info:{Style.RESET_ALL}"
        for inventor in self.inventor_info:
            s += f"\n\t- {inventor['first_name']} {inventor['last_name']} ({inventor['country']}): {inventor['city']}, {inventor['state'] if inventor['state'] else 'N/A'}"

        s += f"\n\n{Style.BRIGHT}Dates Info:{Style.RESET_ALL}"
        for d_name, d_val in self.dates_info.items():
            s += f"\n\t- {d_name}: {d_val}"

        s += f"\n\n{Style.BRIGHT}Classifications:{Style.RESET_ALL}"
        s += f"\n{Style.BRIGHT}IPC:{Style.RESET_ALL}"
        for dic in self.ipc:
            s += f"\n\t- {dic['section']}.{dic['class']}.{dic['subclass']}.{dic['main_group']}/{dic['subgroup']}"
        s += f"\n{Style.BRIGHT}CPC:{Style.RESET_ALL}"
        for dic in self.cpc:
            s += f"\n\t- {dic['section']}.{dic['class']}.{dic['subclass']}.{dic['main_group']}/{dic['subgroup']}"

        s += f"\n\n{Style.BRIGHT}Application Info:{Style.RESET_ALL}"
        s += f"\n\t- Number: {self.application_number}"
        s += f"\n\t- Kind: {self.document_kind}"
        s += f"\n\t- Type: {self.application_type}"

        s += f"\n\n{Style.BRIGHT}Has Abstract:{Style.RESET_ALL}{Fore.GREEN if self.abstract else Fore.RED} {self.abstract is not None}{Style.RESET_ALL}"
        s += f"\t|\t{Style.BRIGHT}Has Claims:{Style.RESET_ALL}{Fore.GREEN if self.claims else Fore.RED} {self.claims is not None}{Style.RESET_ALL}"
        s += f"\t|\t{Style.BRIGHT}Has Description:{Style.RESET_ALL}{Fore.GREEN if self.description else Fore.RED} {self.description is not None}{Style.RESET_ALL}"

        return s

    def to_sqlite(self, conn: sqlite3.Connection) -> None:
        """
        Inserts this object into a SQLite database.

        Args:
            conn (sqlite3.Connection): The SQLite database connection.
        """
        query = """
        INSERT INTO patents (
            patent_id, 
            patent_name, 
            assignee_info, 
            inventor_info, 
            dates_info, 
            classifications, 
            application_number, 
            document_kind, 
            application_type, 
            abstract, 
            claims, 
            description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        data = (
            self.patent_id, 
            self.patent_name, 
            self.assignee_info, 
            self.inventor_info, 
            self.dates_info, 
            self.classifications, 
            self.application_number, 
            self.document_kind, 
            self.application_type, 
            self.abstract, 
            self.claims, 
            self.description
        )
        cursor = conn.cursor()
        try:
            cursor.execute(query, data)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    @classmethod
    def from_sqlite_by_id(cls, conn: sqlite3.Connection, patent_id: int):
        """
        Creates a Patent object from a SQLite database by its patent ID.

        Args:
            conn (sqlite3.Connection): The SQLite database connection.
            patent_id (int): The patent ID to search for.

        Returns:
            Patent: The Patent object created from the database.
        """
        query = f"SELECT * FROM patents WHERE patent_id = ?;"
        cursor = conn.cursor()
        cursor.execute(query, (patent_id,))
        tupl = cursor.fetchone()
        if tupl is None:
            return None
        return cls.from_sqlite(tupl)

    @classmethod
    def from_sqlite(cls, tupl: tuple):
        """
        Creates a Patent object from a tuple of data from a SQLite database.

        Args:
            tupl (tuple): The tuple of data from the database.

        Returns:
            Patent: The Patent object created from the database
        """
        patent_id, patent_name, assignee_info, inventor_info, dates_info, classifications, application_number, document_kind, application_type, abstract, claims, description = tupl
        return cls(patent_id, patent_name, assignee_info, inventor_info, dates_info, classifications, application_number, document_kind, application_type, abstract, claims, description)



