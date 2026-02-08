"""
DBpedia data downloader and parser.
Downloads DBpedia dumps and parses RDF/Turtle format.
"""

import os
import bz2
import requests
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
from pathlib import Path


class DBpediaDownloader:
    """"""
    
    DBPEDIA_VERSION = "2016-04"
    DBPEDIA_BASE_URL = f"https://downloads.dbpedia.org/{DBPEDIA_VERSION}/core-i18n/en/"
    
    REQUIRED_DATASETS = {
        'mappingbased_objects': (f'{DBPEDIA_BASE_URL}mappingbased_objects_en.ttl.bz2', 'mappingbased_objects_en.ttl.bz2'),
        'instance_types': (f'{DBPEDIA_BASE_URL}instance_types_en.ttl.bz2', 'instance_types_en.ttl.bz2'),
        'labels': (f'{DBPEDIA_BASE_URL}labels_en.ttl.bz2', 'labels_en.ttl.bz2'),
        'short_abstracts': (f'{DBPEDIA_BASE_URL}short_abstracts_en.ttl.bz2', 'short_abstracts_en.ttl.bz2'),
    }
    
    def __init__(self, dump_dir: str = "dataset/dbpedia_dumps"):
        """
        Args:
        """
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
        """
        Args:
        Returns:
        """
        try:
            print(f"Downloading from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
                    
            print(f"✓ Downloaded to {dest_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    def download_all_dumps(self, force: bool = False) -> Dict[str, Path]:
        """
        Args:
        Returns:
        """
        downloaded_files = {}
        
        print(f"DBpedia Downloader - Version {self.DBPEDIA_VERSION}")
        print(f"Download directory: {self.dump_dir.absolute()}")
        print(f"Datasets to download: {len(self.REQUIRED_DATASETS)}")
        print("-" * 60)
        
        for dataset_name, (url, filename) in self.REQUIRED_DATASETS.items():
            dest_path = self.dump_dir / filename
            
            if dest_path.exists() and not force:
                file_size = dest_path.stat().st_size / (1024**3)  # GB
                print(f"✓ {dataset_name}: Already exists ({file_size:.2f} GB)")
                downloaded_files[dataset_name] = dest_path
                continue
            
            success = self.download_file(url, dest_path)
            
            if success:
                downloaded_files[dataset_name] = dest_path
            else:
                print(f"Warning: Failed to download {dataset_name}")
            
            time.sleep(1)
        
        print("-" * 60)
        print(f"Download complete: {len(downloaded_files)}/{len(self.REQUIRED_DATASETS)} files")
        
        return downloaded_files
    
    def decompress_file(self, bz2_path: Path, force: bool = False) -> Optional[Path]:
        """
        Args:
        Returns:
        """
        ttl_path = bz2_path.with_suffix('')
        
        if ttl_path.exists() and not force:
            print(f"✓ Already decompressed: {ttl_path.name}")
            return ttl_path
        
        print(f"Decompressing {bz2_path.name}...")
        
        try:
            file_size = bz2_path.stat().st_size
            
            with bz2.open(bz2_path, 'rb') as source, \
                 open(ttl_path, 'wb') as dest, \
                 tqdm(total=file_size, unit='iB', unit_scale=True, desc="Decompressing") as pbar:
                
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    dest.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"✓ Decompressed to {ttl_path}")
            return ttl_path
            
        except Exception as e:
            print(f"✗ Failed to decompress {bz2_path}: {e}")
            if ttl_path.exists():
                ttl_path.unlink()
            return None
    
    def decompress_all(self, force: bool = False) -> Dict[str, Path]:
        """
        Args:
        Returns:
        """
        decompressed_files = {}
        
        print("Decompressing DBpedia dumps...")
        print("-" * 60)
        
        for dataset_name, (url, filename) in self.REQUIRED_DATASETS.items():
            bz2_path = self.dump_dir / filename
            
            if not bz2_path.exists():
                print(f"✗ {dataset_name}: Compressed file not found")
                continue
            
            ttl_path = self.decompress_file(bz2_path, force)
            
            if ttl_path:
                decompressed_files[dataset_name] = ttl_path
        
        print("-" * 60)
        print(f"Decompression complete: {len(decompressed_files)} files")
        
        return decompressed_files
    
    def get_file_stats(self) -> Dict[str, Any]:
        """"""
        stats = {
            'downloaded': {},
            'decompressed': {},
            'total_size_gb': 0
        }
        
        for dataset_name, (url, filename) in self.REQUIRED_DATASETS.items():
            bz2_path = self.dump_dir / filename
            ttl_path = bz2_path.with_suffix('')
            
            if bz2_path.exists():
                size_gb = bz2_path.stat().st_size / (1024**3)
                stats['downloaded'][dataset_name] = {
                    'path': str(bz2_path),
                    'size_gb': round(size_gb, 2)
                }
                stats['total_size_gb'] += size_gb
            
            if ttl_path.exists():
                size_gb = ttl_path.stat().st_size / (1024**3)
                stats['decompressed'][dataset_name] = {
                    'path': str(ttl_path),
                    'size_gb': round(size_gb, 2)
                }
        
        stats['total_size_gb'] = round(stats['total_size_gb'], 2)
        
        return stats
    
    def print_stats(self):
        """"""
        stats = self.get_file_stats()
        
        print("\nDBpedia Dumps Statistics")
        print("=" * 60)
        
        print(f"\nDownloaded files: {len(stats['downloaded'])}/{len(self.REQUIRED_DATASETS)}")
        for name, info in stats['downloaded'].items():
            print(f"  • {name}: {info['size_gb']} GB")
        
        print(f"\nDecompressed files: {len(stats['decompressed'])}/{len(self.REQUIRED_DATASETS)}")
        for name, info in stats['decompressed'].items():
            print(f"  • {name}: {info['size_gb']} GB")
        
        print(f"\nTotal storage: {stats['total_size_gb']} GB")
        print("=" * 60)


def main():
    """"""
    downloader = DBpediaDownloader()
    
    downloader.print_stats()
    
    print("\nThis will download ~5-10GB of DBpedia dumps.")
    response = input("Continue? [y/N]: ")
    
    if response.lower() == 'y':
        downloaded = downloader.download_all_dumps()
        
        if downloaded:
            print("\nDecompressing files...")
            decompressed = downloader.decompress_all()
            
            print(f"\n✓ Ready to use: {len(decompressed)} datasets")
    else:
        print("Download cancelled.")


if __name__ == "__main__":
    main()
