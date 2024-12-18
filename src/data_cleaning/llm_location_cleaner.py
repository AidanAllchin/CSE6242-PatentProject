#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created: Thu Nov 07 2024
Author: Aidan Allchin

This script uses Ollama with qwen2.5:32b to clean and correct location data
from our patent tsv file. It processes locations with missing coordinates due 
to misspellings, typos, etc. and saves a best guess for the correct state, 
city, and country. The mapping between original state, city, and country and
the corrected state, city, and country is saved to a TSV file.

NOTE: DO NOT RUN THIS WITHOUT A POWERFUL MACHINE. Will light your processor on 
fire and take up an extra 20GB of storage.

I also realize this is an insane way of trying to fix this, but if the USPTO 
API continues to not work, this is my best idea.
"""
import os, sys
from pathlib import Path

# ALL FILES IN THE PROJECT SHOULD BE IMPORTED RELATIVE TO THE PROJECT ROOT
# the # of `.parent`s should be adjusted based on the file's location
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style
import pandas as pd
import json
import time
import asyncio
from datetime import datetime
import ollama
from typing import List, Dict, Tuple, Optional
from src.other.logging import PatentLogger
from src.other.helpers import local_filename

# Initialize logger
logger = PatentLogger.get_logger(__name__)

# Constants
BATCH_SIZE      = 10    # may need to lower this if accuracy is atrocious
PREFERRED_MODEL = "qwen2.5:32b"#"llama3.2-vision:11b"
SAVE_INTERVAL   = 1     # Save to TSV every N batches

# Global variables for model state
is_loaded    = False
loaded_model = None


###############################################################################
#                                                                             #
#  ProcessingMetrics: Helper class for tracking timing and processing stats   #
#                                                                             #
###############################################################################


class ProcessingMetrics:
    # This is more so I can keep track of how long this is taking and whether
    # it's even worth attempting/if I should hire this out to runpod or something
    def __init__(self):
        self.start_time          = datetime.now()
        self.batch_times         = []
        self.location_times      = []
        self.total_locations     = 0
        self.completed_locations = 0
        
    def add_batch_time(self, duration: float, batch_size: int):
        """
        Records timing for a batch.
        
        Args:
            duration: Time taken to process the batch
            batch_size: Number of locations in the batch
        """
        self.batch_times.append(duration)
        self.location_times.extend([duration/batch_size] * batch_size)
        self.completed_locations += batch_size
        
    def get_stats(self) -> dict:
        """
        Generate a dictionary of processing metrics.

        Returns:
            dict: Dictionary of processing metrics with keys:
                - runtime: Total runtime of the process
                - avg_batch_time: Average time per batch
                - avg_location_time: Average time per location
                - completion: Percentage of locations processed
                - est_remaining: Estimated time remaining
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        est_remaining = (elapsed/self.completed_locations * (self.total_locations-self.completed_locations)) if self.completed_locations else 0
        if est_remaining > 3600:
            est_remaining = f"{est_remaining/3600:.1f}h"
        elif est_remaining > 60:
            est_remaining = f"{est_remaining/60:.1f}m"

        if elapsed > 3600:
            elapsed = f"{elapsed/3600:.1f}h"
        elif elapsed > 60:
            elapsed = f"{elapsed/60:.1f}m"
        else:
            elapsed = f"{elapsed:.1f}s"
        return {
            'runtime': elapsed,
            'avg_batch_time': f"{sum(self.batch_times)/len(self.batch_times):.2f}s" if self.batch_times else "N/A",
            'avg_location_time': f"{sum(self.location_times)/len(self.location_times):.2f}s" if self.location_times else "N/A",
            'completion': f"{self.completed_locations}/{self.total_locations} ({(self.completed_locations/self.total_locations*100):.1f}%)" if self.total_locations else "0/0 (0%)",
            'est_remaining': f"{est_remaining}"
        }
    
    def print_progress(self):
        """
        Print current metrics in a clean format.
        """
        stats = self.get_stats()
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Style.BRIGHT}{Fore.CYAN}Location Processing Metrics:{Style.RESET_ALL}")
        print(f"Runtime: {Style.BRIGHT}{stats['runtime']}{Style.RESET_ALL}")
        print(f"Average batch time: {Style.BRIGHT}{stats['avg_batch_time']}{Style.RESET_ALL}")
        print(f"Average per location: {Style.BRIGHT}{stats['avg_location_time']}{Style.RESET_ALL}")
        print(f"Progress: {Style.BRIGHT}{stats['completion']}{Style.RESET_ALL}")
        print(f"Estimated remaining: {Style.BRIGHT}{stats['est_remaining']}{Style.RESET_ALL}")


###############################################################################
#                                                                             #
#     LocationCleaner: Main class for cleaning location data using Ollama     #
#                                                                             #
###############################################################################


class LocationCleaner:
    def __init__(self, input_tsv: str, output_tsv: str, diagnostics: bool = False):
        """
        Initialize the LocationCleaner with input and output file paths.
        
        Args:
            input_tsv: Path to the input TSV containing failed locations
            output_tsv: Path to save the corrected mappings
        """
        self.input_tsv = input_tsv
        self.output_tsv = output_tsv
        self.corrections = {} # Running list of corrections
        self.load_existing_corrections()
        self.client = ollama.AsyncClient()
        if diagnostics:
            self.metrics = ProcessingMetrics()
        else: self.metrics = None

        enabled = "enabled" if diagnostics else "disabled"
        logger.info(f"Initialized LocationCleaner with input: {local_filename(input_tsv)}")
        logger.info(f"Output: {local_filename(output_tsv)}")
        logger.info(f"Metrics = {enabled}")

    def load_model(self, model_name: str = PREFERRED_MODEL) -> bool:
        """
        Load the specified model if it's not already loaded.
        Modified from Sky code.

        Args:
            model_name: Name of the model to load

        Returns:
            bool: True if the model was loaded successfully, False otherwise
        """
        global is_loaded, loaded_model

        all_ollama_models = ollama.list()
        all_ollama_models = [model['name'] for model in all_ollama_models['models']]

        logger.info(f"Available models: {all_ollama_models}")

        # Pull model if not already downloaded
        if model_name not in all_ollama_models:
            logger.error(f"Model not found: {model_name}. Attempting to pull it.")
            logger.warning("Depending on your download speed, this can take multiple minutes.")
            try:
                ollama.pull(model_name)
                logger.info(f"Model {model_name} pulled successfully.")
            except Exception as e:
                logger.exception(f"Failed to pull model: {model_name}: {e}")
                return False

        logger.info(f"Model {model_name} loaded successfully.")

        is_loaded = True
        loaded_model = model_name
        return True

    def load_existing_corrections(self):
        """
        Load any existing corrections from the output TSV.
        """
        if os.path.exists(self.output_tsv):
            df = pd.read_csv(self.output_tsv, sep='\t')
            self.corrections = {
                (row['bad_city'], row['bad_state'], row['bad_country']): 
                (row['new_city'], row['new_state'], row['new_country'])
                for _, row in df.iterrows()
            }
            logger.info(f"Loaded {len(self.corrections)} existing entries from the output TSV.")
        else:
            logger.info(f"No existing corrections found in {local_filename(self.output_tsv)}.")

    def save_corrections(self):
        """
        Take a guess what this one does, genius.
        """
        df = pd.DataFrame([
            {
                'bad_city': bad[0], 'bad_state': bad[1], 'bad_country': bad[2],
                'new_city': new[0], 'new_state': new[1], 'new_country': new[2]
            }
            for bad, new in self.corrections.items()
        ])
        df.to_csv(self.output_tsv, sep='\t', index=False)
        logger.info(f"Saved {len(self.corrections)} corrections to {local_filename(self.output_tsv)}.")

    def create_system_prompt(self) -> str:
        """
        Create the system prompt for the model.

        Returns:
            str: The system prompt for the model
        """
        return """You are a location data spell-checking assistant specializing in correcting city, state, and country names.
    Your task is to normalize location data by fixing:
    1. Capitalization (e.g., "new york" -> "New York")
    2. Clear typos (e.g., "chicgo" -> "Chicago")
    3. Special character issues (e.g., "sãn josé" -> "San Jose")

    Rules:
    - Only make corrections when you are absolutely confident
    - Preserve abbreviated state codes (e.g., "CA" stays "CA")
    - Return original values if unsure
    - Never guess at similar-sounding cities
    - Always return 3-element arrays for locations: [city, state, country]
    - Country codes must be 2 letters (e.g., "US", "GB", "DE")
    - Return exact input if any uncertainty exists
    - If one of the values is unknown, return the original value even if the other values have been corrected (i.e. ['seatttle', 'wa', 'unknown'] -> ['Seattle', 'WA', 'unknown'])
    - Focus on low character-change corrections. For example, changing "ato, nj" to "Atco, NJ" is better than "Atlantic City, NJ"
    - The order in the 'corrected' field must be city, state, country, even if the input is in a different order
    - DO NOT CORRECT LOCATIONS TO THE MOST POPULAR CITY/STATE/COUNTRY. ONLY CORRECT BASED ON POTENTIAL TYPOGRAPHICAL ERRORS, MISPELLINGS, OR OTHER OBVIOUS MISTAKES.

    You must respond with a JSON array where each object has exactly two keys:
    - 'original': [city, state, country]
    - 'corrected': [city, state, country]"""
    
    def create_user_prompt(self, locations: List[Tuple[str, str, str]]) -> str:
        """
        Create the user prompt for the batch of locations.
        
        Args:
            locations: List of locations to correct

        Returns:
            str: The user prompt for the batch of locations
        """
        locations_str = "\n".join([
            f"[{json.dumps(city)}, {json.dumps(state)}, {json.dumps(country)}]"
            for city, state, country in locations
        ])
        
        prompt = f"""Please analyze and correct these {len(locations)} location entries.
For each location, provide spelling corrections and proper capitalization.
Only make corrections when absolutely certain, otherwise return the original values.
Be sure to put the state and country in the correct order.

Input locations (as [city, state, country]):
{locations_str}

Respond with a JSON array containing objects with 'original' and 'corrected' arrays for each location.
Example response format:
{{
    locations: [
        {{
            "original": ["new yoork", "ny", "us"],
            "corrected": ["New York", "NY", "US"]
        }},
        {{
            "original": ["sãn josé", "ca", "us"],
            "corrected": ["San Jose", "CA", "US"]
        }},
        {{
            "original": ["seatttlle", "us", "wa"],
            "corrected": ["Seattle", "WA", "US"]
        }}
    ]
}}
"""
        return prompt
    
    async def get_corrections_async(self, locations: List[Tuple[str, str, str]], model_name: str = PREFERRED_MODEL, temp: float = 0.0) -> Optional[Dict]:
        """
        Get corrections for a batch of locations using the async Ollama client.
        Adapted from Sky code.

        Args:
            locations: List of locations to correct
            model_name: Name of the model to use
            temp: Temperature for the model (keeping this at 0.0 is the best move here)

        Returns:
            Dict: Mapping of original locations to corrected locations:
                {
                    (city, state, country): (corrected_city, corrected_state, corrected_country)
                }
        """
        global is_loaded, loaded_model

        if not is_loaded or model_name != loaded_model:
            logger.warning(f"Model {model_name} not loaded. Attempting to load it.")
            if not self.load_model(model_name):
                logger.error(f"Failed to load model: {model_name}.")
                return None

        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": self.create_user_prompt(locations)}
        ]
        
        try:
            response = await self.client.chat(
                model=model_name,
                messages=messages,
                format="json",
                options={"temperature": temp}
            )

            try:
                corrections = response['message']['content']
                # Ensure it starts with the list characters
                corrections = corrections.strip()
                corrections = json.loads(corrections)
                if 'locations' not in corrections:
                    logger.error(f"Unexpected response format: {corrections}")
                    return {loc: loc for loc in locations}
                corrections = corrections['locations']

                result = {}
                
                if isinstance(corrections, list):
                    for correction in corrections:
                        if not isinstance(correction, dict):
                            logger.warning(f"Skipping invalid correction format: {correction}")
                            continue
                            
                        original  = correction.get('original')
                        corrected = correction.get('corrected')
                        
                        #log(f"[internal] {original} -> {corrected}")
                        
                        if (isinstance(original, list) and len(original) == 3 and 
                            isinstance(corrected, list) and len(corrected) == 3):
                            result[tuple(str(x).lower() for x in original)] = tuple(corrected)
                        else:
                            logger.warning(f"Skipping invalid correction format: {correction}")
                
                else:
                    logger.error(f"Unexpected response format: {corrections}")

                # For locations not handled, store lowercase -> lowercase
                for loc in locations:
                    # annoying but have to lowercase and the LLM saves as none instead of nan
                    loc_lower_nan  = tuple(str(x).lower().replace('none', 'nan') for x in loc)
                    loc_lower_none = tuple(str(x).lower().replace('nan', 'none') for x in loc)

                    if loc_lower_nan not in result and loc_lower_none not in result:
                        result[loc_lower_nan] = loc_lower_nan

                return result
            except json.JSONDecodeError:
                logger.exception(f"Failed to decode JSON response: {response['message']['content']}")
                return {loc: loc for loc in locations}

        except Exception as e:
            logger.exception(f"Error in Ollama generation: {e}")
            return {loc: loc for loc in locations}
        
    async def process_locations(self):
        """
        Process all locations with missing coordinates.
        
        This is the main function that processes all locations with missing
        coordinates in the input TSV file. It iterates through the locations
        in batches, sends them to the Ollama API, then saves the corrected
        mappings to the output file.

        Also responsible for maintaining the processing metrics and saving
        the corrections at regular intervals.
        """
        st = time.time()
        df = pd.read_csv(self.input_tsv, sep='\t')

        failed_locations = df[
            (df['latitude'] == 0.0) & 
            (df['longitude'] == 0.0)
        ][['city', 'state', 'country']].drop_duplicates().values.tolist()
        
        locations_to_process = [
            tuple(loc) for loc in failed_locations 
            if tuple(loc) not in self.corrections
        ]
        logger.info(f"Loaded {len(failed_locations)} unique locations with missing coordinates in {time.time()-st:.2f}s.")
        
        if self.metrics:
            self.metrics.total_locations = len(locations_to_process)
        
        # Chunk into batches
        for i in range(0, len(locations_to_process), BATCH_SIZE):
            batch_start = time.time()
            batch = locations_to_process[i:i + BATCH_SIZE]
            
            # Get corrections for the batch
            batch_corrections = await self.get_corrections_async(batch)
            if batch_corrections:
                # Update the corrections
                # print("CURRENT CORRECTIONS:")
                # print(self.corrections)
                # print("BATCH CORRECTIONS:")
                # print(batch_corrections)
                self.corrections.update(batch_corrections)
                # print("\nUPDATED CORRECTIONS:")
                # print(self.corrections)
            
            batch_time = time.time() - batch_start
            if self.metrics:
                self.metrics.add_batch_time(batch_time, len(batch))
            
            if i % (BATCH_SIZE * SAVE_INTERVAL) == 0:
                self.save_corrections()
                if self.metrics:
                    self.metrics.print_progress()
            
            await asyncio.sleep(0.05)
        
        # Final stats and save
        if self.metrics:
            self.metrics.print_progress()
        self.save_corrections()

async def display_metrics_task(metrics: ProcessingMetrics):
    """
    Display processing metrics at regular intervals (every second) while not
    blocking the LLM process.

    Args:
        metrics: ProcessingMetrics object to display
    """
    while True:
        metrics.print_progress()
        await asyncio.sleep(1)

async def main():
    logger.info(f"\nStarting local-LLM location cleaner.")
    logger.warning("DO NOT RUN THIS ON A LAPTOP.")

    i = input("Seriously - are you sure you want to continue? (y/n): ")
    if i.lower() != 'y':
        logger.error("Exiting.")
        return

    cleaner = LocationCleaner(
        input_tsv  = os.path.join(project_root, "data", "geolocation", "city_coordinates.tsv"),
        output_tsv = os.path.join(project_root, "data", "geolocation", "location_corrections.tsv"),
        diagnostics=True
    )

    # Start displaying timing stuff
    metrics_task = asyncio.create_task(display_metrics_task(cleaner.metrics))
    await cleaner.process_locations()
    metrics_task.cancel()

    try:
        await metrics_task
    except asyncio.CancelledError:
        logger.info("Metrics display task cancelled.")

async def test():
    cleaner = LocationCleaner(
        input_tsv  = os.path.join(project_root, "data", "geolocation", "city_coordinates.tsv"),
        output_tsv = os.path.join(project_root, "data", "geolocation", "location_corrections.tsv"),
        diagnostics=False
    )
    demo_locations = [
        ("new yoork", "ny", "us"),
        ("sãn josé", "ca", "us"),
        #("ardvark", "zz", "us")
    ]
    corrections = await cleaner.get_corrections_async(demo_locations)
    for original, corrected in corrections.items():
        print(f"  - {original} -> {corrected}")

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(test())