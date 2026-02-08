"""
DCI-Compliant Payload Structure Design Module

This module implements the 48-bit payload structure for forensic marking:
- 16-bit timestamp (increments every 15 minutes, resets yearly)
- 20-bit location/sequence ID (FMID) - supports 1,048,576 unique positions
- 12-bit error correction code (CRC-12)

The payload is designed to be extractable from any 5-minute video segment.
"""

import numpy as np
from typing import Tuple
import datetime


class DCIPayload:
    """
    DCI-compliant payload structure for forensic marking.
    
    Total: 48 bits
    - Timestamp: 16 bits (0-65535, increments every 15 minutes, yearly reset)
    - FMID: 20 bits (0-1048575, unique forensic marking ID)
    - ECC: 12 bits (CRC-12 error correction)
    """
    
    # CRC-12 polynomial: x^12 + x^11 + x^3 + x^2 + x + 1
    CRC12_POLY = 0x80F  # 100000001111 in binary
    
    def __init__(self, fmid: int):
        """
        Initialize payload encoder/decoder with a forensic marking ID.
        
        Args:
            fmid: Forensic Marking ID (0-1048575, 20 bits)
        """
        if not 0 <= fmid <= 0xFFFFF:  # 20-bit range
            raise ValueError(f"FMID must be in range [0, 1048575], got {fmid}")
        self.fmid = fmid
    
    @staticmethod
    def calculate_timestamp(dt: datetime.datetime = None) -> int:
        """
        Calculate 16-bit timestamp from datetime.
        
        The timestamp increments every 15 minutes and resets yearly.
        This gives us 365.25 * 24 * 4 = 35,064 possible values per year,
        which fits comfortably in 16 bits (0-65535).
        
        Args:
            dt: datetime object (defaults to current time)
            
        Returns:
            16-bit timestamp value (0-65535)
        """
        if dt is None:
            dt = datetime.datetime.now(datetime.timezone.utc)
        
        # Get day of year (1-366)
        day_of_year = dt.timetuple().tm_yday
        
        # Calculate minutes since start of day
        minutes_today = dt.hour * 60 + dt.minute
        
        # Number of 15-minute intervals since start of year
        # (day_of_year - 1) because tm_yday starts at 1
        intervals_since_year_start = (day_of_year - 1) * 96 + (minutes_today // 15)
        
        # Ensure it fits in 16 bits
        timestamp = intervals_since_year_start & 0xFFFF
        
        return timestamp
    
    @staticmethod
    def timestamp_to_datetime(timestamp: int, year: int = None) -> Tuple[int, int, int]:
        """
        Convert 16-bit timestamp back to day/hour/minute.
        
        Args:
            timestamp: 16-bit timestamp value
            year: Optional year for context
            
        Returns:
            Tuple of (day_of_year, hour, minute)
        """
        if year is None:
            year = datetime.datetime.now(datetime.timezone.utc).year
        
        # Each day has 96 intervals (24 * 60 / 15)
        day_of_year = (timestamp // 96) + 1
        remaining_intervals = timestamp % 96
        
        # Convert intervals to hour and minute
        total_minutes = remaining_intervals * 15
        hour = total_minutes // 60
        minute = total_minutes % 60
        
        return day_of_year, hour, minute
    
    @staticmethod
    def compute_crc12(data: int, data_bits: int = 36) -> int:
        """
        Compute CRC-12 checksum for error detection.
        
        Args:
            data: Input data (36 bits: 16-bit timestamp + 20-bit FMID)
            data_bits: Number of bits in data
            
        Returns:
            12-bit CRC checksum
        """
        # Initialize CRC register
        crc = 0
        
        # Process each bit
        for i in range(data_bits - 1, -1, -1):
            bit = (data >> i) & 1
            msb = (crc >> 11) & 1
            
            crc = ((crc << 1) | bit) & 0xFFF
            
            if msb:
                crc ^= DCIPayload.CRC12_POLY & 0xFFF
        
        # Process 12 more zero bits to flush the register
        for _ in range(12):
            msb = (crc >> 11) & 1
            crc = (crc << 1) & 0xFFF
            if msb:
                crc ^= DCIPayload.CRC12_POLY & 0xFFF
        
        return crc
    
    @staticmethod
    def verify_crc12(data: int, crc: int, data_bits: int = 36) -> bool:
        """
        Verify CRC-12 checksum.
        
        Args:
            data: Original data (36 bits)
            crc: Received CRC (12 bits)
            data_bits: Number of bits in data
            
        Returns:
            True if CRC is valid, False otherwise
        """
        computed_crc = DCIPayload.compute_crc12(data, data_bits)
        return computed_crc == crc
    
    def encode_payload(self, timestamp: int = None) -> np.ndarray:
        """
        Encode a complete 48-bit payload.
        
        Args:
            timestamp: Optional 16-bit timestamp (auto-calculated if None)
            
        Returns:
            NumPy array of 48 bits (0s and 1s)
        """
        if timestamp is None:
            timestamp = self.calculate_timestamp()
        
        # Validate timestamp
        if not 0 <= timestamp <= 0xFFFF:
            raise ValueError(f"Timestamp must be 16-bit (0-65535), got {timestamp}")
        
        # Combine timestamp and FMID into 36-bit data
        data_36bit = (timestamp << 20) | self.fmid
        
        # Calculate CRC-12
        crc = self.compute_crc12(data_36bit)
        
        # Combine into 48-bit payload
        payload_48bit = (data_36bit << 12) | crc
        
        # Convert to binary array
        bits = np.zeros(48, dtype=np.int32)
        for i in range(48):
            bits[47 - i] = (payload_48bit >> i) & 1
        
        return bits
    
    @staticmethod
    def decode_payload(bits: np.ndarray) -> Tuple[int, int, int, bool]:
        """
        Decode a 48-bit payload.
        
        Args:
            bits: NumPy array of 48 bits
            
        Returns:
            Tuple of (timestamp, fmid, crc, is_valid)
        """
        if len(bits) != 48:
            raise ValueError(f"Expected 48 bits, got {len(bits)}")
        
        # Convert binary array to integer
        payload_48bit = 0
        for i, bit in enumerate(bits):
            payload_48bit = (payload_48bit << 1) | int(bit > 0.5)
        
        # Extract components
        crc = payload_48bit & 0xFFF  # Last 12 bits
        data_36bit = payload_48bit >> 12  # First 36 bits
        
        timestamp = (data_36bit >> 20) & 0xFFFF  # First 16 bits of data
        fmid = data_36bit & 0xFFFFF  # Last 20 bits of data
        
        # Verify CRC
        is_valid = DCIPayload.verify_crc12(data_36bit, crc)
        
        return timestamp, fmid, crc, is_valid
    
    def encode_for_vidstamp(self, timestamp: int = None) -> np.ndarray:
        """
        Encode payload in VidStamp-compatible format.
        
        VidStamp expects a 100-bit input (56 data bits + 44 BCH ECC bits).
        We'll use our 48-bit payload and pad with zeros, or use the first 
        48 bits of the 100-bit space.
        
        For simplicity and to maintain compatibility, we embed our 48-bit
        payload in the first 48 bits and pad the rest.
        
        Args:
            timestamp: Optional 16-bit timestamp
            
        Returns:
            NumPy array of 100 bits for VidStamp compatibility
        """
        # Get our 48-bit payload
        payload_bits = self.encode_payload(timestamp)
        
        # Create 100-bit array (VidStamp format)
        vidstamp_bits = np.zeros(100, dtype=np.int32)
        vidstamp_bits[:48] = payload_bits
        
        return vidstamp_bits
    
    @staticmethod
    def decode_from_vidstamp(bits: np.ndarray) -> Tuple[int, int, int, bool]:
        """
        Decode payload from VidStamp output.
        
        Args:
            bits: NumPy array of 100 bits from VidStamp
            
        Returns:
            Tuple of (timestamp, fmid, crc, is_valid)
        """
        if len(bits) < 48:
            raise ValueError(f"Expected at least 48 bits, got {len(bits)}")
        
        # Extract our 48-bit payload
        payload_bits = bits[:48]
        
        return DCIPayload.decode_payload(payload_bits)


def generate_test_payload():
    """Generate a test payload with current timestamp."""
    # Example FMID for screen #12345
    fmid = 12345
    payload = DCIPayload(fmid)
    
    # Generate payload with current timestamp
    bits = payload.encode_payload()
    
    # Decode to verify
    timestamp, decoded_fmid, crc, is_valid = DCIPayload.decode_payload(bits)
    
    print(f"Generated Payload:")
    print(f"  FMID: {decoded_fmid} (0x{decoded_fmid:05X})")
    print(f"  Timestamp: {timestamp} (0x{timestamp:04X})")
    print(f"  CRC-12: {crc} (0x{crc:03X})")
    print(f"  Valid: {is_valid}")
    
    # Convert timestamp to date/time
    day, hour, minute = DCIPayload.timestamp_to_datetime(timestamp)
    print(f"  Time: Day {day} at {hour:02d}:{minute:02d}")
    
    return bits


if __name__ == "__main__":
    # Run test
    print("DCI Payload Structure Test\n" + "="*50)
    bits = generate_test_payload()
    print(f"\nPayload bits: {bits}")
    print(f"Payload (hex): {''.join(str(b) for b in bits)}")
