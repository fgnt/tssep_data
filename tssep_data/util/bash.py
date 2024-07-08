import os
import getpass
import socket
from pathlib import Path

class c:  # noqa
    Color_Off = '\033[0m'  # Text Reset
    Black = '\033[0;30m'  # Black
    Red = '\033[0;31m'  # Red
    Green = '\033[0;32m'  # Green
    Yellow = '\033[0;33m'  # Yellow
    Blue = '\033[0;34m'  # Blue
    Purple = '\033[0;35m'  # Purple
    Cyan = '\033[0;36m'  # Cyan
    White = '\033[0;37m'  # White



def USER(): return getpass.getuser()  # bash: $USER

def hostname(): return socket.gethostname()  # bash: hostname

_fqdn = socket.getfqdn()  # Full hostname
_PC2SYSNAME = os.environ.get('PC2SYSNAME')

if _PC2SYSNAME == 'Noctua2':
    location = 'Noctua2'  # Paderborn Cluster
elif _PC2SYSNAME == 'Noctua1':
    location = 'Noctua1'  # Paderborn Cluster
elif _fqdn.endswith('.cm.cluster'):
    # ToDo: Does this work?
    #       A fallback would be to check if /mm1/ exists.
    location = 'MERL'
else:
    location = 'unknown'
