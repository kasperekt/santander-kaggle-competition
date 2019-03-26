import getpass

username = getpass.getuser()

# Use it only on server
use_gpu = username == 'paperspace'

# Jobs
n_jobs = 8 if username == 'paperspace' else 4