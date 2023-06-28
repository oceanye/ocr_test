import tempfile
import os
import subprocess
...
temp_dir = tempfile.mkdtemp()
temp_file = os.path.join(temp_dir, 'eng.traineddata')
...
command = ['combine_tessdata', '-o', temp_file, 'listfile.txt']
subprocess.run(command, check=True)
