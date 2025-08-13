"""
Universal Binary Principle (UBP) Framework v2.0 - HexDictionary Module

This module implements the HexDictionary Universal Data Layer, providing
efficient hexadecimal-based data storage and retrieval that integrates
seamlessly with the UBP framework's OffBit structure.

Based on the Hex Dictionary v2 dataset concept, this system stores data
in a highly compressed, searchable format that can represent not just
words but any structured data within the UBP computational space.

Author: Euan Craig
Version: 2.0
Date: August 2025
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
import json
import hashlib
import struct
import zlib
from collections import defaultdict
import pickle

from .core import UBPConstants
from .bitfield import Bitfield, OffBit


@dataclass
class HexEntry:
    """A single entry in the HexDictionary."""
    hex_key: str
    data_type: str
    raw_data: Any
    compressed_data: bytes
    metadata: Dict[str, Any]
    access_count: int
    creation_timestamp: float
    last_access_timestamp: float


@dataclass
class HexDictionaryStats:
    """Statistics for HexDictionary performance and usage."""
    total_entries: int
    total_size_bytes: int
    compression_ratio: float
    average_access_time: float
    cache_hit_rate: float
    most_accessed_keys: List[str]
    data_type_distribution: Dict[str, int]


class HexDictionary:
    """
    Universal Data Layer using hexadecimal-based efficient storage.
    
    This class provides a high-performance data storage and retrieval system
    that can handle various data types (strings, numbers, OffBits, arrays)
    and compress them efficiently using hexadecimal encoding schemes.
    """
    
    def __init__(self, max_cache_size: int = 10000, compression_level: int = 6):
        """
        Initialize the HexDictionary.
        
        Args:
            max_cache_size: Maximum number of entries to keep in memory cache
            compression_level: Compression level for data storage (1-9)
        """
        self.entries: Dict[str, HexEntry] = {}
        self.cache: Dict[str, Any] = {}
        self.max_cache_size = max_cache_size
        self.compression_level = compression_level
        
        # Performance tracking
        self.access_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Data type handlers
        self.type_handlers = {
            'string': self._handle_string,
            'integer': self._handle_integer,
            'float': self._handle_float,
            'offbit': self._handle_offbit,
            'array': self._handle_array,
            'bitfield_coords': self._handle_bitfield_coords,
            'json': self._handle_json,
            'binary': self._handle_binary
        }
        
        # Reverse lookup indices
        self.data_type_index: Dict[str, Set[str]] = defaultdict(set)
        self.metadata_index: Dict[str, Set[str]] = defaultdict(set)
        
        print("✅ UBP HexDictionary Universal Data Layer Initialized")
        print(f"   Max Cache Size: {max_cache_size:,}")
        print(f"   Compression Level: {compression_level}")
        print(f"   Supported Data Types: {list(self.type_handlers.keys())}")
    
    def _generate_hex_key(self, data: Any, data_type: str) -> str:
        """
        Generate a unique hexadecimal key for the given data.
        
        Args:
            data: Data to generate key for
            data_type: Type of the data
            
        Returns:
            Hexadecimal string key
        """
        # Create a hash of the data and type
        data_str = str(data) + data_type
        hash_obj = hashlib.sha256(data_str.encode('utf-8'))
        
        # Take first 16 characters of hex digest for efficiency
        hex_key = hash_obj.hexdigest()[:16]
        
        # Ensure uniqueness by checking existing keys
        counter = 0
        original_key = hex_key
        while hex_key in self.entries:
            counter += 1
            hex_key = f"{original_key}_{counter:04x}"
        
        return hex_key
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        return zlib.compress(data, level=self.compression_level)
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data using zlib."""
        return zlib.decompress(compressed_data)
    
    # ========================================================================
    # DATA TYPE HANDLERS
    # ========================================================================
    
    def _handle_string(self, data: str) -> bytes:
        """Handle string data type."""
        return data.encode('utf-8')
    
    def _handle_integer(self, data: int) -> bytes:
        """Handle integer data type."""
        return struct.pack('>q', data)  # Big-endian 64-bit signed integer
    
    def _handle_float(self, data: float) -> bytes:
        """Handle float data type."""
        return struct.pack('>d', data)  # Big-endian 64-bit double
    
    def _handle_offbit(self, data: int) -> bytes:
        """Handle OffBit data type."""
        # Store OffBit as 32-bit integer with layer information
        layers = OffBit.get_all_layers(data)
        layer_bytes = struct.pack('>IBBBB', data, 
                                 layers['reality'], layers['information'],
                                 layers['activation'], layers['unactivated'])
        return layer_bytes
    
    def _handle_array(self, data: Union[List, np.ndarray]) -> bytes:
        """Handle array data type."""
        if isinstance(data, np.ndarray):
            return data.tobytes()
        else:
            # Convert list to numpy array and then to bytes
            array = np.array(data)
            return array.tobytes()
    
    def _handle_bitfield_coords(self, data: Tuple[int, ...]) -> bytes:
        """Handle Bitfield coordinates."""
        if len(data) != 6:
            raise ValueError("Bitfield coordinates must be 6-dimensional")
        return struct.pack('>6I', *data)
    
    def _handle_json(self, data: Dict[str, Any]) -> bytes:
        """Handle JSON-serializable data."""
        json_str = json.dumps(data, sort_keys=True)
        return json_str.encode('utf-8')
    
    def _handle_binary(self, data: bytes) -> bytes:
        """Handle raw binary data."""
        return data
    
    # ========================================================================
    # REVERSE DATA TYPE HANDLERS
    # ========================================================================
    
    def _restore_string(self, data: bytes) -> str:
        """Restore string from bytes."""
        return data.decode('utf-8')
    
    def _restore_integer(self, data: bytes) -> int:
        """Restore integer from bytes."""
        return struct.unpack('>q', data)[0]
    
    def _restore_float(self, data: bytes) -> float:
        """Restore float from bytes."""
        return struct.unpack('>d', data)[0]
    
    def _restore_offbit(self, data: bytes) -> int:
        """Restore OffBit from bytes."""
        offbit_value, reality, info, activation, unactivated = struct.unpack('>IBBBB', data)
        return offbit_value
    
    def _restore_array(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Restore array from bytes."""
        dtype = metadata.get('dtype', 'float64')
        shape = metadata.get('shape', (-1,))
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    
    def _restore_bitfield_coords(self, data: bytes) -> Tuple[int, ...]:
        """Restore Bitfield coordinates from bytes."""
        return struct.unpack('>6I', data)
    
    def _restore_json(self, data: bytes) -> Dict[str, Any]:
        """Restore JSON data from bytes."""
        json_str = data.decode('utf-8')
        return json.loads(json_str)
    
    def _restore_binary(self, data: bytes) -> bytes:
        """Restore binary data."""
        return data
    
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    def store(self, data: Any, data_type: str, metadata: Optional[Dict[str, Any]] = None,
              custom_key: Optional[str] = None) -> str:
        """
        Store data in the HexDictionary.
        
        Args:
            data: Data to store
            data_type: Type of data (must be in supported types)
            metadata: Optional metadata dictionary
            custom_key: Optional custom hex key (if None, auto-generated)
            
        Returns:
            Hexadecimal key for the stored data
        """
        import time
        start_time = time.time()
        
        if data_type not in self.type_handlers:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Generate or use custom key
        if custom_key:
            if custom_key in self.entries:
                raise ValueError(f"Key {custom_key} already exists")
            hex_key = custom_key
        else:
            hex_key = self._generate_hex_key(data, data_type)
        
        # Handle the data based on its type
        handler = self.type_handlers[data_type]
        raw_bytes = handler(data)
        
        # Add metadata for arrays
        if metadata is None:
            metadata = {}
        
        if data_type == 'array' and isinstance(data, np.ndarray):
            metadata['dtype'] = str(data.dtype)
            metadata['shape'] = data.shape
        
        # Compress the data
        compressed_bytes = self._compress_data(raw_bytes)
        
        # Create entry
        entry = HexEntry(
            hex_key=hex_key,
            data_type=data_type,
            raw_data=data,  # Keep original for cache
            compressed_data=compressed_bytes,
            metadata=metadata,
            access_count=0,
            creation_timestamp=time.time(),
            last_access_timestamp=time.time()
        )
        
        # Store entry
        self.entries[hex_key] = entry
        
        # Update indices
        self.data_type_index[data_type].add(hex_key)
        for key, value in metadata.items():
            self.metadata_index[f"{key}:{value}"].add(hex_key)
        
        # Add to cache
        self._update_cache(hex_key, data)
        
        # Record access time
        access_time = time.time() - start_time
        self.access_times.append(access_time)
        
        return hex_key
    
    def retrieve(self, hex_key: str) -> Any:
        """
        Retrieve data from the HexDictionary.
        
        Args:
            hex_key: Hexadecimal key of the data
            
        Returns:
            Retrieved data in its original form
        """
        import time
        start_time = time.time()
        
        # Check cache first
        if hex_key in self.cache:
            self.cache_hits += 1
            self._update_access_stats(hex_key)
            return self.cache[hex_key]
        
        self.cache_misses += 1
        
        # Retrieve from storage
        if hex_key not in self.entries:
            raise KeyError(f"Key {hex_key} not found in HexDictionary")
        
        entry = self.entries[hex_key]
        
        # Decompress data
        raw_bytes = self._decompress_data(entry.compressed_data)
        
        # Restore data based on type
        restore_method_name = f"_restore_{entry.data_type}"
        if hasattr(self, restore_method_name):
            restore_method = getattr(self, restore_method_name)
            if entry.data_type == 'array':
                data = restore_method(raw_bytes, entry.metadata)
            else:
                data = restore_method(raw_bytes)
        else:
            raise ValueError(f"No restore method for data type: {entry.data_type}")
        
        # Update cache
        self._update_cache(hex_key, data)
        
        # Update access statistics
        self._update_access_stats(hex_key)
        
        # Record access time
        access_time = time.time() - start_time
        self.access_times.append(access_time)
        
        return data
    
    def _update_cache(self, hex_key: str, data: Any) -> None:
        """Update the memory cache with new data."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.entries[k].last_access_timestamp)
            del self.cache[oldest_key]
        
        self.cache[hex_key] = data
    
    def _update_access_stats(self, hex_key: str) -> None:
        """Update access statistics for an entry."""
        import time
        entry = self.entries[hex_key]
        entry.access_count += 1
        entry.last_access_timestamp = time.time()
    
    def delete(self, hex_key: str) -> bool:
        """
        Delete an entry from the HexDictionary.
        
        Args:
            hex_key: Key to delete
            
        Returns:
            True if deleted, False if key not found
        """
        if hex_key not in self.entries:
            return False
        
        entry = self.entries[hex_key]
        
        # Remove from indices
        self.data_type_index[entry.data_type].discard(hex_key)
        for key, value in entry.metadata.items():
            self.metadata_index[f"{key}:{value}"].discard(hex_key)
        
        # Remove from cache
        if hex_key in self.cache:
            del self.cache[hex_key]
        
        # Remove entry
        del self.entries[hex_key]
        
        return True
    
    def exists(self, hex_key: str) -> bool:
        """Check if a key exists in the dictionary."""
        return hex_key in self.entries
    
    def get_entry_info(self, hex_key: str) -> Dict[str, Any]:
        """
        Get detailed information about an entry.
        
        Args:
            hex_key: Key to get info for
            
        Returns:
            Dictionary with entry information
        """
        if hex_key not in self.entries:
            raise KeyError(f"Key {hex_key} not found")
        
        entry = self.entries[hex_key]
        
        return {
            'hex_key': entry.hex_key,
            'data_type': entry.data_type,
            'metadata': entry.metadata,
            'access_count': entry.access_count,
            'creation_timestamp': entry.creation_timestamp,
            'last_access_timestamp': entry.last_access_timestamp,
            'compressed_size_bytes': len(entry.compressed_data),
            'in_cache': hex_key in self.cache
        }
    
    # ========================================================================
    # SEARCH AND QUERY OPERATIONS
    # ========================================================================
    
    def find_by_type(self, data_type: str) -> List[str]:
        """
        Find all keys of a specific data type.
        
        Args:
            data_type: Data type to search for
            
        Returns:
            List of hex keys
        """
        return list(self.data_type_index.get(data_type, set()))
    
    def find_by_metadata(self, metadata_key: str, metadata_value: Any) -> List[str]:
        """
        Find all keys with specific metadata.
        
        Args:
            metadata_key: Metadata key to search for
            metadata_value: Metadata value to match
            
        Returns:
            List of hex keys
        """
        search_key = f"{metadata_key}:{metadata_value}"
        return list(self.metadata_index.get(search_key, set()))
    
    def search(self, query: Dict[str, Any]) -> List[str]:
        """
        Search for entries matching multiple criteria.
        
        Args:
            query: Dictionary with search criteria
                   Supported keys: 'data_type', 'metadata', 'min_access_count'
                   
        Returns:
            List of hex keys matching all criteria
        """
        result_keys = set(self.entries.keys())
        
        # Filter by data type
        if 'data_type' in query:
            type_keys = set(self.find_by_type(query['data_type']))
            result_keys &= type_keys
        
        # Filter by metadata
        if 'metadata' in query:
            for key, value in query['metadata'].items():
                metadata_keys = set(self.find_by_metadata(key, value))
                result_keys &= metadata_keys
        
        # Filter by access count
        if 'min_access_count' in query:
            min_count = query['min_access_count']
            access_filtered = {k for k in result_keys 
                             if self.entries[k].access_count >= min_count}
            result_keys &= access_filtered
        
        return list(result_keys)
    
    # ========================================================================
    # UBP INTEGRATION METHODS
    # ========================================================================
    
    def store_offbit_collection(self, offbits: List[int], 
                               collection_name: str) -> str:
        """
        Store a collection of OffBits with UBP-specific metadata.
        
        Args:
            offbits: List of OffBit values
            collection_name: Name for the collection
            
        Returns:
            Hex key for the stored collection
        """
        # Calculate collection statistics
        coherence_values = [OffBit.calculate_coherence(ob) for ob in offbits]
        active_count = sum(1 for ob in offbits if OffBit.is_active(ob))
        
        metadata = {
            'collection_name': collection_name,
            'count': len(offbits),
            'active_count': active_count,
            'average_coherence': np.mean(coherence_values) if coherence_values else 0.0,
            'max_coherence': max(coherence_values) if coherence_values else 0.0,
            'min_coherence': min(coherence_values) if coherence_values else 0.0,
            'ubp_collection': True
        }
        
        return self.store(offbits, 'array', metadata)
    
    def store_bitfield_region(self, bitfield: Bitfield, 
                            start_coords: Tuple[int, ...], 
                            end_coords: Tuple[int, ...],
                            region_name: str) -> str:
        """
        Store a region of a Bitfield.
        
        Args:
            bitfield: Bitfield instance
            start_coords: Starting coordinates (6D)
            end_coords: Ending coordinates (6D)
            region_name: Name for the region
            
        Returns:
            Hex key for the stored region
        """
        if len(start_coords) != 6 or len(end_coords) != 6:
            raise ValueError("Coordinates must be 6-dimensional")
        
        # Extract region data
        region_data = []
        region_coords = []
        
        # Simple extraction (in practice, would be more sophisticated)
        for i in range(start_coords[0], min(end_coords[0] + 1, bitfield.dimensions[0])):
            for j in range(start_coords[1], min(end_coords[1] + 1, bitfield.dimensions[1])):
                for k in range(start_coords[2], min(end_coords[2] + 1, bitfield.dimensions[2])):
                    for l in range(start_coords[3], min(end_coords[3] + 1, bitfield.dimensions[3])):
                        for m in range(start_coords[4], min(end_coords[4] + 1, bitfield.dimensions[4])):
                            for n in range(start_coords[5], min(end_coords[5] + 1, bitfield.dimensions[5])):
                                coords = (i, j, k, l, m, n)
                                offbit = bitfield.get_offbit(coords)
                                if offbit != 0:  # Only store non-zero OffBits
                                    region_data.append(offbit)
                                    region_coords.append(coords)
        
        metadata = {
            'region_name': region_name,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'bitfield_dimensions': bitfield.dimensions,
            'non_zero_count': len(region_data),
            'coordinates': region_coords,
            'ubp_bitfield_region': True
        }
        
        return self.store(region_data, 'array', metadata)
    
    def create_ubp_index(self, index_name: str, 
                        key_extractor: callable) -> Dict[str, List[str]]:
        """
        Create a custom index for UBP-specific data.
        
        Args:
            index_name: Name of the index
            key_extractor: Function that extracts index key from entry data
            
        Returns:
            Dictionary mapping index keys to hex keys
        """
        index = defaultdict(list)
        
        for hex_key, entry in self.entries.items():
            if entry.metadata.get('ubp_collection') or entry.metadata.get('ubp_bitfield_region'):
                try:
                    index_key = key_extractor(entry.raw_data, entry.metadata)
                    index[index_key].append(hex_key)
                except Exception:
                    continue  # Skip entries that can't be indexed
        
        # Store the index itself
        index_dict = dict(index)
        self.store(index_dict, 'json', {'index_name': index_name, 'ubp_index': True})
        
        return index_dict
    
    # ========================================================================
    # STATISTICS AND MAINTENANCE
    # ========================================================================
    
    def get_statistics(self) -> HexDictionaryStats:
        """Get comprehensive statistics about the HexDictionary."""
        total_size = sum(len(entry.compressed_data) for entry in self.entries.values())
        
        if self.entries:
            # Calculate compression ratio
            total_raw_size = sum(len(pickle.dumps(entry.raw_data)) for entry in self.entries.values())
            compression_ratio = total_size / max(1, total_raw_size)
            
            # Most accessed keys
            sorted_entries = sorted(self.entries.items(), 
                                  key=lambda x: x[1].access_count, reverse=True)
            most_accessed = [key for key, _ in sorted_entries[:10]]
            
            # Data type distribution
            type_dist = {}
            for entry in self.entries.values():
                type_dist[entry.data_type] = type_dist.get(entry.data_type, 0) + 1
        else:
            compression_ratio = 1.0
            most_accessed = []
            type_dist = {}
        
        # Cache hit rate
        total_accesses = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(1, total_accesses)
        
        # Average access time
        avg_access_time = np.mean(self.access_times) if self.access_times else 0.0
        
        return HexDictionaryStats(
            total_entries=len(self.entries),
            total_size_bytes=total_size,
            compression_ratio=compression_ratio,
            average_access_time=avg_access_time,
            cache_hit_rate=cache_hit_rate,
            most_accessed_keys=most_accessed,
            data_type_distribution=type_dist
        )
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage by recompressing data and cleaning up indices.
        
        Returns:
            Dictionary with optimization results
        """
        original_size = sum(len(entry.compressed_data) for entry in self.entries.values())
        
        # Recompress all entries with maximum compression
        recompressed_count = 0
        for entry in self.entries.values():
            original_compressed = entry.compressed_data
            raw_bytes = self._decompress_data(original_compressed)
            new_compressed = zlib.compress(raw_bytes, level=9)  # Maximum compression
            
            if len(new_compressed) < len(original_compressed):
                entry.compressed_data = new_compressed
                recompressed_count += 1
        
        # Clean up empty index entries
        for index_dict in [self.data_type_index, self.metadata_index]:
            empty_keys = [k for k, v in index_dict.items() if not v]
            for k in empty_keys:
                del index_dict[k]
        
        new_size = sum(len(entry.compressed_data) for entry in self.entries.values())
        size_reduction = original_size - new_size
        
        return {
            'original_size_bytes': original_size,
            'new_size_bytes': new_size,
            'size_reduction_bytes': size_reduction,
            'size_reduction_percent': (size_reduction / max(1, original_size)) * 100,
            'recompressed_entries': recompressed_count,
            'total_entries': len(self.entries)
        }
    
    def export_to_file(self, filename: str) -> None:
        """Export the entire HexDictionary to a file."""
        export_data = {
            'entries': {},
            'metadata': {
                'version': '2.0',
                'total_entries': len(self.entries),
                'export_timestamp': time.time()
            }
        }
        
        # Export all entries
        for hex_key, entry in self.entries.items():
            export_data['entries'][hex_key] = {
                'hex_key': entry.hex_key,
                'data_type': entry.data_type,
                'compressed_data': entry.compressed_data.hex(),  # Convert to hex string
                'metadata': entry.metadata,
                'access_count': entry.access_count,
                'creation_timestamp': entry.creation_timestamp,
                'last_access_timestamp': entry.last_access_timestamp
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ HexDictionary exported to {filename}")
        print(f"   Entries: {len(self.entries):,}")
        print(f"   File size: {os.path.getsize(filename):,} bytes")
    
    @classmethod
    def import_from_file(cls, filename: str) -> 'HexDictionary':
        """Import a HexDictionary from a file."""
        with open(filename, 'r') as f:
            import_data = json.load(f)
        
        # Create new instance
        hex_dict = cls()
        
        # Import all entries
        for hex_key, entry_data in import_data['entries'].items():
            # Convert hex string back to bytes
            compressed_data = bytes.fromhex(entry_data['compressed_data'])
            
            entry = HexEntry(
                hex_key=entry_data['hex_key'],
                data_type=entry_data['data_type'],
                raw_data=None,  # Will be loaded on demand
                compressed_data=compressed_data,
                metadata=entry_data['metadata'],
                access_count=entry_data['access_count'],
                creation_timestamp=entry_data['creation_timestamp'],
                last_access_timestamp=entry_data['last_access_timestamp']
            )
            
            hex_dict.entries[hex_key] = entry
            
            # Rebuild indices
            hex_dict.data_type_index[entry.data_type].add(hex_key)
            for key, value in entry.metadata.items():
                hex_dict.metadata_index[f"{key}:{value}"].add(hex_key)
        
        print(f"✅ HexDictionary imported from {filename}")
        print(f"   Entries: {len(hex_dict.entries):,}")
        
        return hex_dict
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the HexDictionary."""
        stats = self.get_statistics()
        
        return {
            'total_entries': stats.total_entries,
            'total_size_mb': stats.total_size_bytes / (1024 * 1024),
            'compression_ratio': stats.compression_ratio,
            'cache_size': len(self.cache),
            'cache_hit_rate': stats.cache_hit_rate,
            'average_access_time_ms': stats.average_access_time * 1000,
            'data_types': list(stats.data_type_distribution.keys()),
            'most_accessed_keys': stats.most_accessed_keys[:5],
            'supported_types': list(self.type_handlers.keys())
        }


if __name__ == "__main__":
    # Test the HexDictionary
    print("="*60)
    print("UBP HEXDICTIONARY MODULE TEST")
    print("="*60)
    
    import time
    import os
    
    # Create HexDictionary
    hex_dict = HexDictionary(max_cache_size=100)
    
    # Test basic data types
    print("\n--- Basic Data Type Tests ---")
    
    # String
    string_key = hex_dict.store("Hello UBP World!", "string", {"category": "greeting"})
    print(f"Stored string with key: {string_key}")
    
    # Integer
    int_key = hex_dict.store(42, "integer", {"category": "answer"})
    print(f"Stored integer with key: {int_key}")
    
    # Float
    float_key = hex_dict.store(3.14159, "float", {"category": "constant"})
    print(f"Stored float with key: {float_key}")
    
    # OffBit
    test_offbit = OffBit.create_offbit(reality=15, information=31, activation=7, unactivated=3)
    offbit_key = hex_dict.store(test_offbit, "offbit", {"category": "ubp_data"})
    print(f"Stored OffBit with key: {offbit_key}")
    
    # Array
    test_array = np.random.rand(100)
    array_key = hex_dict.store(test_array, "array", {"category": "simulation_data"})
    print(f"Stored array with key: {array_key}")
    
    # Test retrieval
    print("\n--- Retrieval Tests ---")
    retrieved_string = hex_dict.retrieve(string_key)
    print(f"Retrieved string: {retrieved_string}")
    
    retrieved_offbit = hex_dict.retrieve(offbit_key)
    print(f"Retrieved OffBit: {retrieved_offbit:032b}")
    
    retrieved_array = hex_dict.retrieve(array_key)
    print(f"Retrieved array shape: {retrieved_array.shape}")
    
    # Test UBP-specific operations
    print("\n--- UBP Integration Tests ---")
    
    # Store OffBit collection
    offbit_collection = [OffBit.create_offbit(reality=i, information=i*2, activation=i%8) 
                        for i in range(10)]
    collection_key = hex_dict.store_offbit_collection(offbit_collection, "test_collection")
    print(f"Stored OffBit collection with key: {collection_key}")
    
    # Test search functionality
    print("\n--- Search Tests ---")
    
    # Find by type
    string_keys = hex_dict.find_by_type("string")
    print(f"String keys: {string_keys}")
    
    # Find by metadata
    greeting_keys = hex_dict.find_by_metadata("category", "greeting")
    print(f"Greeting keys: {greeting_keys}")
    
    # Complex search
    ubp_data = hex_dict.search({"metadata": {"category": "ubp_data"}})
    print(f"UBP data keys: {ubp_data}")
    
    # Test performance
    print("\n--- Performance Tests ---")
    
    # Store many entries
    start_time = time.time()
    for i in range(100):
        key = hex_dict.store(f"test_string_{i}", "string", {"batch": "performance_test"})
    store_time = time.time() - start_time
    print(f"Stored 100 strings in {store_time:.4f} seconds")
    
    # Retrieve many entries
    performance_keys = hex_dict.find_by_metadata("batch", "performance_test")
    start_time = time.time()
    for key in performance_keys:
        data = hex_dict.retrieve(key)
    retrieve_time = time.time() - start_time
    print(f"Retrieved {len(performance_keys)} strings in {retrieve_time:.4f} seconds")
    
    # Get statistics
    print("\n--- Statistics ---")
    stats = hex_dict.get_statistics()
    print(f"Total entries: {stats.total_entries}")
    print(f"Total size: {stats.total_size_bytes:,} bytes")
    print(f"Compression ratio: {stats.compression_ratio:.3f}")
    print(f"Cache hit rate: {stats.cache_hit_rate:.3f}")
    print(f"Average access time: {stats.average_access_time*1000:.3f} ms")
    print(f"Data type distribution: {stats.data_type_distribution}")
    
    # Test optimization
    print("\n--- Optimization Test ---")
    optimization_results = hex_dict.optimize_storage()
    print(f"Size reduction: {optimization_results['size_reduction_percent']:.2f}%")
    print(f"Recompressed entries: {optimization_results['recompressed_entries']}")
    
    # Get final status
    status = hex_dict.get_status()
    print(f"\nHexDictionary Status:")
    print(f"  Total entries: {status['total_entries']:,}")
    print(f"  Total size: {status['total_size_mb']:.2f} MB")
    print(f"  Cache hit rate: {status['cache_hit_rate']:.3f}")
    print(f"  Supported types: {len(status['supported_types'])}")
    
    print("\n✅ HexDictionary module test completed successfully!")

