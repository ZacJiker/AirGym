import socket
import struct

UDP_READ_BUFFER_SIZE = 16384
UDP_WRITE_BUFFER_SIZE = 16384

class ErrorUDPWriteSize(Exception):
    """Raised when the plugin is unable to write the specified number of bytes."""
    pass

class DataRefError(Exception):
    """Raised when the plugin is unable to write the specified number of bytes."""
    pass

class XPlaneConnect(object):
    """XPlaneConnect (XPC) facilitates communication to and from the XPCPlugin."""

    def __init__(self, hostname: str ='localhost', xplane_port: int = 49009, client_port: int = 0, timeout: int = 100):
        """Sets up a new connection to an X-Plane Connect plugin running in X-Plane.
            Args:
              hostname: The hostname of the machine running X-Plane.
              xplane_port: The port on which the XPC plugin is listening. Usually 49007.
              client_port: The port which will be used to send and receive data.
              timeout: The period (in milliseconds) after which read attempts will fail.
        """

        # Validate parameters
        try:
            xplane_ip_adress = socket.gethostbyname(hostname)
        except:
            raise ValueError("Unable to resolve hostname.")

        if xplane_port < 0 or xplane_port > 65535:
            raise ValueError("The specified X-Plane port is not a valid port number.")
        if client_port < 0 or client_port > 65535:
            raise ValueError("The specified Client port is not a valid port number.")
        if timeout < 0:
            raise ValueError("Timeout must be non-negative.")

        # Setup XPlane IP and port
        self.xplane_dest = (xplane_ip_adress, xplane_port)

        # Create and bind socket
        client_addr = ("0.0.0.0", client_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.bind(client_addr)
        timeout /= 1000.0
        self.socket.settimeout(timeout)

    def __del__(self):
        self.socket.close()

    # Define __enter__ and __exit__ to allow use in a with statement

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def close(self):
        """Closes the connection to the XPC plugin."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    # Define read and write UDP methods

    def _read_udp(self) -> bytes:
        """Reads a specified number of bytes from the X-Plane Connect plugin.
        
            Returns:
              The bytes read from the plugin.
        """
        return self.socket.recv(UDP_READ_BUFFER_SIZE)
    
    def _write_udp(self, data: bytes):
        """Writes a specified number of bytes to the X-Plane Connect plugin.
        
            Args:
              data: The bytes to write to the plugin.
        """
        if len(data) > UDP_WRITE_BUFFER_SIZE or len(data) == 0:
            raise ErrorUDPWriteSize("The specified data is too large to write to the plugin.")
        
        self.socket.sendto(data, self.xplane_dest)

    # Define methods for sending and receiving data

    def send_dref(self, dref: str, value: float):
        """Sets the value of a single dataref.
        
            Args:
              dref: The name of the dataref to set.
              value: The value to set the dataref to.
        """
        buffer = struct.pack(b"<4sx", b"DREF")

        if len(dref) > 255 or len(dref) == 0:
            raise DataRefError()

        if hasattr(value, "__len__"):
            # If value is a list, then we need to send the length of the list
            fmt = "<B{0:d}sB{1:d}f".format(len(dref), len(value))
            buffer += struct.pack(fmt.encode(), len(dref), dref.encode(), len(value), value)
        else:
            fmt = "<B{0:d}sBf".format(len(dref))
            buffer += struct.pack(fmt.encode(), len(dref), dref.encode(), 1, value)

        self._write_udp(buffer)

    def get_dref(self, dref: str) -> float:
        """Gets the value of a single dataref.
        
            Args:
              dref: The name of the dataref to get.
            
            Returns:
              The value of the specified dataref.
        """
        # Send the request
        buffer = struct.pack(b"<4sxB", b"GETD", len(dref))
        fmt = "<B{0:d}s".format(len(dref))
        buffer += struct.pack(fmt.encode(), len(dref), dref.encode())
        self._write_udp(buffer)

        # Read the response and parse it
        data = self._read_udp()
        row_length = struct.unpack_from(b"B", data)[0]
    
        fmt = "<{0:d}f".format(row_length)
        row = struct.unpack_from(fmt.encode(), buffer, 1)