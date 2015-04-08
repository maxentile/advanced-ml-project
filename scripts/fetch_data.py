# Download data from: "Single-Cell Mass Cytometry of Differential Immune and
# Drug Responses Across a Human Hematopoietic Continuum"
# Hosted on Cytobank
url = 'http://www.cytobank.org/nolanlab/cytof_bone_marrow_signaling_data.acs'
target = '../data/cytof_bone_marrow_signaling_data.acs'

def main():
    import os.path

    if os.path.isfile(target):
        print('file already downloaded')
    else:
        print('downloading file: this will take a while')

        import urllib
        opener = urllib.request.URLopener()
        opener.retrieve(url,target)

if __name__ == '__main__':
    main()
