import tarfile
import numpy as np
import sys


tar = tarfile.open("/home/ruth/ProgrammingProjects/AQUS/AQUAS/data/oa_comm_xml.PMC009xxxxxx.baseline.2022-12-18.tar.gz", "r:gz")
for member in tar.getmembers():
     f = tar.extractfile(member)
     if f is not None:
         content = f.read()
         print("%s has %d characters" % (member, len(content)))
         print(content)
         sys.exit()

tar.close()