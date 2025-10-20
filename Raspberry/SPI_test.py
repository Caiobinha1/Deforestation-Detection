#This code sets up SPI communication between a Raspberry Pi (master) and an STM32 microcontroller (slave), for testing purposes.
import spidev
import time
import array as arr # Import array module for bytearray conversion

# SPI bus (0 or 1 on Raspberry Pi, typically 0)
bus = 0
# Chip select pin (CE0 or CE1, typically 0 for CE0)
device = 0

# Initialize SPI
spi = spidev.SpiDev()
spi.open(bus, device) # Open a connection to a specific bus and device [2, 26]

# Configure SPI settings (these MUST match the STM32 slave's configuration)
spi.max_speed_hz = 1000000  # Example speed: 1 MHz (adjust as needed, ensure STM32 can handle) [26]
spi.mode = 0                # Example: SPI Mode 0 (CPOL=0, CPHA=0). Match STM32 setting! [16, 26]
# spi.bits_per_word = 8     # Default for spidev, but good to be explicit if needed [26]

try:
    while True:
        # Data to send from Raspberry Pi (list of 8-bit integers)
        # Ensure all values are within 0-255 range for 8-bit transfer
        tx_data = [0x10]
        
        # Perform a full-duplex SPI transaction.
        # spi.xfer2() sends `tx_data` and simultaneously receives data into `rx_data`.
        # Chip Select (NSS) is held active for the entire transaction with xfer2().[26]
        rx_data = spi.xfer2(tx_data)
        
        print(f"Sent: {[hex(b) for b in tx_data]}, Received: {[hex(b) for b in rx_data]}")
        time.sleep(1) # Wait for 1 second before next transaction

except KeyboardInterrupt:
    spi.close() # Close the SPI connection cleanly [26]
    print("SPI communication stopped.")