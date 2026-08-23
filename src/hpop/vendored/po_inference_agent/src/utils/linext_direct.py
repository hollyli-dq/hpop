"""
Direct linext C++ library interface using ctypes.
Provides 100x speedup without subprocess overhead.
"""

import ctypes
import numpy as np
import os
from typing import Optional
from .silence_cpp_output import QuietLinextMixin,silence_cpp_output

class LinextDirect(QuietLinextMixin):
    """
    Direct interface to linext C++ library via ctypes.
    Only uses C++ library - no Python fallback.
    """
    
    def __init__(self, library_path: Optional[str] = None, quiet: bool = True):
        """Initialize direct linext library interface."""
        self.lib = None
        self.available = False
        self._quiet_mode = quiet
        
        if library_path is None:
            # Look for compiled shared library
            possible_paths = [
                './linext/liblinext.dll',      # Windows DLL (current dir)
                './linext/liblinext.so',       # Linux shared library
                './linext/liblinext.dylib',    # macOS shared library  
                '../linext/liblinext.dll',     # Windows DLL (parent dir)
                '../linext/liblinext.so',
                '../linext/liblinext.dylib',
                '../../linext/liblinext.dll',  # Windows DLL (grandparent dir)
                '../../linext/liblinext.so',
                '../../linext/liblinext.dylib',
                'linext/liblinext.dll',        # Windows DLL (relative)
                'linext/liblinext.so',         # Linux (relative)
                'linext/liblinext.dylib',      # macOS (relative)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    library_path = path
                    break
        
        if library_path and os.path.exists(library_path):
            try:
                # Load the shared library
                self.lib = ctypes.CDLL(library_path)
                
                # Define function signature for flattened matrix interface
                # long count_linear_extensions_flat(int* matrix_flat, int size)
                self.lib.count_linear_extensions_flat.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int
                ]
                self.lib.count_linear_extensions_flat.restype = ctypes.c_long
                
                # Log-space version (never overflows)
                # double count_linear_extensions_log_flat(int* matrix_flat, int size)
                self.lib.count_linear_extensions_log_flat.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int
                ]
                self.lib.count_linear_extensions_log_flat.restype = ctypes.c_double
                
                self.available = True
                if not quiet:
                    print(f"✅ LinextDirect C++ library loaded: {library_path}")
                
            except Exception as e:
                if not quiet:
                    print(f"❌ Failed to load linext library: {e}")
                self.available = False
        else:
            # No shared library available
            self.available = False
            if not quiet:
                print("⚠️  LinextDirect C++ library not found - only BasicUtils.nle available")
            
    def nle(self, adj_matrix: np.ndarray) -> int:
        """
        Compute NLE using direct C++ library call.
        Raises RuntimeError if C++ library is not available.
        """
        if not self.available:
            raise RuntimeError("LinextDirect C++ library not available. Use BasicUtils.nle instead.")
            
        if adj_matrix.size == 0 or len(adj_matrix.shape) != 2:
            return 1
            
        if adj_matrix.shape[0] <= 1:
            return 1
            
        def _compute_nle():
            try:
                # Convert numpy array to flattened C array
                n = adj_matrix.shape[0]
                matrix_flat = adj_matrix.astype(np.int32).flatten()
                c_array = (ctypes.c_int * len(matrix_flat))(*matrix_flat)
                
                # Call C++ function directly
                result = self.lib.count_linear_extensions_flat(c_array, n)
                
                if result < 0:  # Error indicator
                    raise RuntimeError(f"C++ library returned error code: {result}")
                    
                return max(1, int(result))
                
            except Exception as e:
                raise RuntimeError(f"C++ library call failed: {e}")
        
        # Use quiet mode if enabled during initialization
        # DISABLED: silence_cpp_output() can cause hangs on Windows
        # Just call directly regardless of quiet mode
        return _compute_nle()
    
    def log_nle(self, adj_matrix: np.ndarray) -> float:
        """
        Compute LOG(NLE) using direct C++ library call.
        This version never overflows - uses log-space arithmetic.
        Returns log of the number of linear extensions.
        """
        if not self.available:
            raise RuntimeError("LinextDirect C++ library not available.")
            
        if adj_matrix.size == 0 or len(adj_matrix.shape) != 2:
            return 0.0  # log(1) = 0
            
        if adj_matrix.shape[0] <= 1:
            return 0.0
            
        try:
            # Convert numpy array to flattened C array
            n = adj_matrix.shape[0]
            matrix_flat = adj_matrix.astype(np.int32).flatten()
            c_array = (ctypes.c_int * len(matrix_flat))(*matrix_flat)
            
            # Call C++ log function directly
            result = self.lib.count_linear_extensions_log_flat(c_array, n)
            
            if result < -1e299:  # Error indicator
                raise RuntimeError(f"C++ library returned error (log)")
                
            return float(result)
            
        except Exception as e:
            raise RuntimeError(f"C++ library log_nle call failed: {e}")

# Global instance
_linext_direct = None

def get_linext_direct(quiet: bool = False) -> LinextDirect:
    """Get singleton LinextDirect instance."""
    global _linext_direct
    if _linext_direct is None:
        _linext_direct = LinextDirect(quiet=quiet)
    return _linext_direct 