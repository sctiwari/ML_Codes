
import os
import subprocess
import sys

if __name__ == "__main__":

    script = sys.argv[1]

    while True:
        print("running")
        subprocess.call(['./manta', script])
