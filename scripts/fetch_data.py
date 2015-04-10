''' Download cytometry data from papers'''

import os.path
import urllib.request

files = []

# 1. Download data from: "Single-Cell Mass Cytometry of Differential Immune and
# Drug Responses Across a Human Hematopoietic Continuum"
# Hosted on Cytobank
url = 'http://www.cytobank.org/nolanlab/cytof_bone_marrow_signaling_data.acs'
target = '../data/cytof_bone_marrow_signaling_data.acs'

files.append((url,target))

# 2. Download data from SPADE paper
# 2.1 simulated fcs file
url = 'http://www.nature.com/nbt/journal/v29/n10/extref/nbt.1991-S2.zip'
target = '../data/spade/simulated_fcs.zip'
files.append((url,target))

# 2.2 Qiu_SPADE_MouseBM.fcs
url = 'http://www.nature.com/nbt/journal/v29/n10/extref/nbt.1991-S3.zip'
target = '../data/spade/Qiu_SPADE_MouseBM.zip'
files.append((url,target))

# 2.2 Qiu_SPADE_MouseBM.fcs
url = 'http://www.nature.com/nbt/journal/v29/n10/extref/nbt.1991-S3.zip'
target = '../data/spade/Qiu_SPADE_MouseBM.zip'
files.append((url,target))


def main():

    for url,target in files:
        if os.path.isfile(target):
            print('file already downloaded')
        else:
            print('downloading file: this will take a while')


            opener = urllib.request.URLopener()
            opener.retrieve(url,target)

if __name__ == '__main__':
    main()
