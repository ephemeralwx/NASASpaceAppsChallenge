import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import re
from pathlib import Path
import pickle
import json
from functools import wraps
import logging
from typing import Optional, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, cache_dir="data"):
        self.base_url = "https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.papers = []
        
        # Setup session with retry logic
        self.session = self._create_session()
        
        # Cache files
        self.csv_cache = self.cache_dir / "publications_base.csv"
        self.abstracts_cache = self.cache_dir / "abstracts_cache.pkl"
        self.metadata_cache = self.cache_dir / "metadata_cache.pkl"
        self.full_data_cache = self.cache_dir / "full_publications.csv"
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry logic and proper headers"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers to look like a real browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        return session
    
    def load_publications(self, force_refresh=False) -> pd.DataFrame:
        """
        Load all 608 NASA space biology publications with complete data.
        
        Args:
            force_refresh: If True, re-fetch all data even if cached
            
        Returns:
            DataFrame with all publications including abstracts
        """
        # Check if we have fully cached data
        if not force_refresh and self.full_data_cache.exists():
            logger.info(f"Loading cached full dataset from {self.full_data_cache}")
            df = pd.read_csv(self.full_data_cache)
            logger.info(f"Loaded {len(df)} publications from cache")
            return df
        
        # Step 1: Load base CSV
        df = self._load_base_csv(force_refresh)
        
        # Step 2: Extract years from titles/links
        df = self._extract_years(df)
        
        # Step 3: Load cached abstracts if available
        abstracts_dict = self._load_cached_abstracts()
        
        # Step 4: Fetch missing abstracts
        df = self._fetch_all_abstracts(df, abstracts_dict)
        
        # Step 5: Extract additional metadata
        df = self._extract_metadata(df)
        
        # Step 6: Save complete dataset
        self._save_full_dataset(df)
        
        logger.info(f"Successfully loaded {len(df)} publications with complete data")
        logger.info(f"Papers with abstracts: {(df['abstract'] != '').sum()}")
        logger.info(f"Average abstract length: {df['abstract'].str.len().mean():.0f} characters")
        
        return df
    
    def _load_base_csv(self, force_refresh=False) -> pd.DataFrame:
        """Load the base CSV from GitHub or cache"""
        if not force_refresh and self.csv_cache.exists():
            logger.info("Loading base CSV from cache")
            return pd.read_csv(self.csv_cache)
        
        logger.info(f"Downloading CSV from {self.base_url}")
        try:
            df = pd.read_csv(self.base_url)
            df.to_csv(self.csv_cache, index=False)
            logger.info(f"Downloaded and cached {len(df)} publications")
            return df
        except Exception as e:
            logger.error(f"Failed to download CSV: {e}")
            if self.csv_cache.exists():
                logger.info("Falling back to cached version")
                return pd.read_csv(self.csv_cache)
            raise
    
    def _load_cached_abstracts(self) -> Dict[str, str]:
        """Load previously fetched abstracts from cache"""
        if self.abstracts_cache.exists():
            logger.info("Loading cached abstracts")
            with open(self.abstracts_cache, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_abstracts_cache(self, abstracts_dict: Dict[str, str]):
        """Save abstracts to cache"""
        with open(self.abstracts_cache, 'wb') as f:
            pickle.dump(abstracts_dict, f)
        logger.info(f"Saved {len(abstracts_dict)} abstracts to cache")
    
    def _fetch_all_abstracts(self, df: pd.DataFrame, abstracts_dict: Dict[str, str]) -> pd.DataFrame:
        """Fetch abstracts for all papers, using cache when available"""
        df['abstract'] = ''
        df['authors'] = ''
        df['abstract_source'] = ''  # Track where abstract came from
        
        total_papers = len(df)
        cached_count = 0
        fetch_count = 0
        failed_count = 0
        
        logger.info(f"Processing {total_papers} papers...")
        
        for idx in tqdm(range(total_papers), desc="Fetching abstracts"):
            row = df.iloc[idx]
            link = row['Link']
            
            # Check cache first
            if link in abstracts_dict and abstracts_dict[link]:
                df.at[idx, 'abstract'] = abstracts_dict[link]['abstract']
                df.at[idx, 'authors'] = abstracts_dict[link].get('authors', '')
                df.at[idx, 'abstract_source'] = 'cache'
                cached_count += 1
                continue
            
            # Fetch from PMC
            try:
                result = self._fetch_article_data(link)
                
                if result and result.get('abstract'):
                    df.at[idx, 'abstract'] = result['abstract']
                    df.at[idx, 'authors'] = result.get('authors', '')
                    df.at[idx, 'abstract_source'] = 'pmc'
                    
                    # Cache the result
                    abstracts_dict[link] = result
                    fetch_count += 1
                    
                    # Save cache periodically (every 50 papers)
                    if fetch_count % 50 == 0:
                        self._save_abstracts_cache(abstracts_dict)
                        logger.info(f"Progress: {fetch_count} new, {cached_count} cached, {failed_count} failed")
                else:
                    # Try alternative methods
                    fallback_abstract = self._try_alternative_sources(row)
                    df.at[idx, 'abstract'] = fallback_abstract
                    df.at[idx, 'abstract_source'] = 'fallback' if fallback_abstract else 'failed'
                    failed_count += 1
                
                # Rate limiting - be respectful to servers
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to fetch abstract for row {idx} ({link}): {e}")
                fallback_abstract = self._try_alternative_sources(row)
                df.at[idx, 'abstract'] = fallback_abstract
                df.at[idx, 'abstract_source'] = 'fallback' if fallback_abstract else 'failed'
                failed_count += 1
                time.sleep(1)  # Longer delay after error
        
        # Final cache save
        self._save_abstracts_cache(abstracts_dict)
        
        logger.info(f"Abstract fetching complete:")
        logger.info(f"  - From cache: {cached_count}")
        logger.info(f"  - Newly fetched: {fetch_count}")
        logger.info(f"  - Failed/Fallback: {failed_count}")
        
        return df
    
    def _fetch_article_data(self, pmc_link: str, timeout=15) -> Optional[Dict]:
        """
        Fetch abstract and metadata from PMC article.
        Returns dict with abstract, authors, etc.
        """
        try:
            response = self.session.get(pmc_link, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            result = {
                'abstract': '',
                'authors': ''
            }
            
            # Method 1: Try structured abstract div
            abstract_section = soup.find('div', {'class': 'abstract'})
            if abstract_section:
                paragraphs = abstract_section.find_all('p')
                if paragraphs:
                    result['abstract'] = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # pmc has inconsistent css classes across years
            if not result['abstract']:
                for class_name in ['tsec', 'abstract-content', 'section abstract']:
                    section = soup.find('div', {'class': class_name})
                    if section:
                        text = section.get_text().strip()
                        # filter out navigation elements that are too short
                        if len(text) > 100:
                            result['abstract'] = text
                            break
            
            if not result['abstract']:
                meta_abstract = soup.find('meta', {'name': 'description'})
                if meta_abstract and meta_abstract.get('content'):
                    result['abstract'] = meta_abstract['content']
            
            if not result['abstract']:
                abstract_heading = soup.find(['h2', 'h3', 'h4'], string=re.compile(r'Abstract', re.I))
                if abstract_heading:
                    abstract_text = []
                    for sibling in abstract_heading.find_next_siblings():
                        if sibling.name in ['h2', 'h3', 'h4']:
                            break
                        if sibling.name == 'p':
                            abstract_text.append(sibling.get_text().strip())
                    if abstract_text:
                        result['abstract'] = ' '.join(abstract_text)
            
            # Extract authors
            author_list = soup.find('div', {'class': 'contrib-group'})
            if author_list:
                authors = author_list.find_all('a', {'class': 'name'})
                if authors:
                    # limit to avoid bloated author lists
                    result['authors'] = ', '.join([a.get_text().strip() for a in authors[:10]])
            
            # Alternative author extraction
            if not result['authors']:
                meta_authors = soup.find('meta', {'name': 'citation_authors'})
                if meta_authors and meta_authors.get('content'):
                    result['authors'] = meta_authors['content']
            
            # Clean up abstract
            if result['abstract']:
                # Remove excessive whitespace
                result['abstract'] = re.sub(r'\s+', ' ', result['abstract']).strip()
                # Prevent memory issues with massive abstracts
                if len(result['abstract']) > 5000:
                    result['abstract'] = result['abstract'][:5000] + '...'
            
            return result if result['abstract'] else None
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request failed for {pmc_link}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Parsing failed for {pmc_link}: {e}")
            return None
    
    def _try_alternative_sources(self, row: pd.Series) -> str:
        """
        Try alternative methods to get abstract when PMC fails.
        This could include DOI lookup, title-based search, etc.
        """
        # hacky fallback - just using title when pmc fails
        # todo: could try pubmed api, crossref, semantic scholar
        title = row['Title']
        return f"[Abstract unavailable - Title]: {title}"
    
    def _extract_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract publication year from various sources"""
        df['year'] = 0
        
        for idx, row in df.iterrows():
            year = None
            
            # Try to extract from title
            year_match = re.search(r'20[0-2][0-9]', str(row['Title']))
            if year_match:
                year = int(year_match.group())
            
            # Try to extract from link/DOI
            if not year and 'Link' in row:
                year_match = re.search(r'20[0-2][0-9]', str(row['Link']))
                if year_match:
                    year = int(year_match.group())
            
            # arbitrary fallback since space biology papers are recent
            df.at[idx, 'year'] = year if year else 2020
        
        return df
    
    def _extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional metadata from abstracts and titles"""
        
        # Extract keywords/topics
        df['has_microgravity'] = df['Title'].str.lower().str.contains('microgravity|weightless', na=False)
        df['has_radiation'] = df['Title'].str.lower().str.contains('radiation', na=False)
        df['has_bone'] = df['Title'].str.lower().str.contains('bone|skeletal', na=False)
        df['has_plant'] = df['Title'].str.lower().str.contains('plant|vegetation', na=False)
        
        # Word count
        df['abstract_word_count'] = df['abstract'].str.split().str.len()
        
        return df
    
    def _save_full_dataset(self, df: pd.DataFrame):
        """Save the complete dataset with all metadata"""
        df.to_csv(self.full_data_cache, index=False)
        logger.info(f"Saved complete dataset to {self.full_data_cache}")
        
        # explicit type conversion because json doesn't handle numpy types
        stats = {
            'total_papers': int(len(df)),
            'papers_with_abstracts': int((df['abstract'] != '').sum()),
            'papers_with_real_abstracts': int((df['abstract_source'] == 'pmc').sum()),
            'average_abstract_length': float(df['abstract'].str.len().mean()),
            'year_range': (int(df['year'].min()), int(df['year'].max())),
            'cached_time': time.time()
        }
        
        stats_file = self.cache_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def clear_cache(self):
        """Clear all cached data - use when you want to refresh everything"""
        cache_files = [
            self.csv_cache,
            self.abstracts_cache,
            self.metadata_cache,
            self.full_data_cache
        ]
        
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Deleted {cache_file}")
        
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached data"""
        stats = {}
        
        if self.abstracts_cache.exists():
            with open(self.abstracts_cache, 'rb') as f:
                abstracts = pickle.load(f)
                stats['cached_abstracts'] = len(abstracts)
        
        if self.full_data_cache.exists():
            df = pd.read_csv(self.full_data_cache)
            stats['total_papers'] = len(df)
            stats['complete_abstracts'] = (df['abstract_source'] == 'pmc').sum()
        
        stats_file = self.cache_dir / 'dataset_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats.update(json.load(f))
        
        return stats