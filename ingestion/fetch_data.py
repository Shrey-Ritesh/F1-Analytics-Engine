import pandas as pd
import fastf1
import fastf1.ergast
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def setup_cache(cache_dir: str):
    """
    Sets up the FastF1 cache directory to avoid re-downloading large datasets.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    logging.info(f"FastF1 cache enabled at {cache_dir}")

def fetch_season_data(year: int, data_dir: str):
    """
    Fetches race results, lap times, pit stops, and circuit/driver info for a given season.
    Saves the combined raw data as CSVs in the specified data directory.
    """
    logging.info(f"Starting data ingestion for season {year}...")
    
    # Store all accumulated data
    all_laps = []
    all_results = []
    
    # Ergast client for basic info (schedule etc.)
    ergast = fastf1.ergast.Ergast()
    
    # Fetch schedule for the year
    schedule = fastf1.get_event_schedule(year)
    
    # We only care about actual Race sessions (skipping testing)
    races = schedule[schedule['EventFormat'] != 'testing']
    
    for _, event in races.iterrows():
        round_num = event['RoundNumber']
        event_name = event['EventName']
        circuit_name = event['EventName'] # Use EventName for circuit to avoid merging multiple races in the same country
        
        # In case some future races in 2025 haven't happened yet, skip if getting session fails
        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(telemetry=False, weather=True, messages=False) # Laps and weather, telemetry takes too long / space
        except Exception as e:
            logging.warning(f"Could not load session {event_name} (Round {round_num}): {e}")
            continue
            
        logging.info(f"Processed session: {year} {event_name}")
        
        # 1. Lap Times (Includes pit stop info, track temp context if merged with weather)
        if session.laps is not None and not session.laps.empty:
            weather_data = session.laps.get_weather_data()
            laps = session.laps.copy()
            laps.reset_index(drop=True, inplace=True)
            weather_data.reset_index(drop=True, inplace=True)
            laps = laps.join(weather_data, lsuffix="_lap", rsuffix="_weather") # Join weather on index
            
            laps['race_year'] = year
            laps['round_number'] = round_num
            laps['event_name'] = event_name
            laps['circuit'] = circuit_name
            all_laps.append(laps)
            
        # 2. Race Results (Includes driver info, finishing positions)
        if session.results is not None and len(session.results) > 0:
            results = session.results.copy()
            results['race_year'] = year
            results['round_number'] = round_num
            results['event_name'] = event_name
            results['circuit'] = circuit_name
            all_results.append(results)

    # Save Laps Data
    if all_laps:
        laps_df = pd.concat(all_laps, ignore_index=True)
        laps_path = os.path.join(data_dir, f'raw_laps_{year}.csv')
        laps_df.to_csv(laps_path, index=False)
        logging.info(f"Saved {len(laps_df)} laps to {laps_path}")
    else:
        logging.warning("No lap data found.")
        
    # Save Results Data
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_path = os.path.join(data_dir, f'raw_results_{year}.csv')
        results_df.to_csv(results_path, index=False)
        logging.info(f"Saved {len(results_df)} results to {results_path}")
    else:
        logging.warning("No results data found.")
        
    logging.info(f"Data ingestion for season {year} complete.")

if __name__ == "__main__":
    # Test execution when script runs directly
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    setup_cache(os.path.join(target_dir, 'cache'))
    fetch_season_data(2025, target_dir)
