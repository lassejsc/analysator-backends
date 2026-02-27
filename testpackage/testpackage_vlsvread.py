import analysator as pt
import numpy as np
import vlsvrs
import os
import hashlib
import pickle

datalocation = "/wrk-vakka/group/spacephysics/vlasiator"
files=["3D/FID/bulk1/bulk1.0000995.vlsv",
       "3D/FHA/bulk1/bulk1.0001165.vlsv",
       "2D/BCQ/bulk/bulk.002002.vlsv"
]
# "2D/BCQ/bulk/bulk.0002002.vlsv",
# filename='/home/siclasse/Downloads/bulk_hermite_compressed.0000001.vlsv'
# filename='/home/siclasse/bulk.0000110.vlsv'
class Tester:
    def __init__(self,filename):
        self.filename=filename
        self.vlsvobj=None
        self.hashes_dict={}
    def changeFile(self,filename):
        self.filename=filename
    def loadPickle(self,file):
        self.pickled = pickle.load(file)
    def dumpPickle(self,file):
        pickle.dump(self.hashes_dict,file)

    def load(self):
        self.vlsvobj_rust=vlsvrs.VlsvFile(self.filename)
        self.vlsvobj_python=pt.vlsvfile.VlsvReader(self.filename)
    def setHashTarget(self,backend):
        if backend=='rust':
            self.vlsvobj=self.vlsvobj_rust
        elif backend=='python':
            self.vlsvobj=self.vlsvobj_python
        else:
            print("None set, give valid backend")
    # def readvlsvrs(self,override=False):
    #     if not self.vlsvobj or override:
    #         self.vlsvobj=vlsvrs.VlsvFile(self.filename)
    #     else:
    #         print("Vlsv already read via vlsvrs or vlsv.")
    #     return 0
    #
    # def readvlsv(self,override=False):
    #     if not self.vlsvobj or override:
    #         self.vlsvobj=pt.vlsvfile.VlsvReader(self.filename)
    #     else:
    #         print("Vlsv already read via vlsvrs or vlsv.")
    #     return 0
    #
    def hash(self,func,args):

        try:
            t=getattr(self.vlsvobj,func)
            retval=t(**args)
        except Exception as e:
            raise e
        self.hashes_dict[func]=hashlib.sha256(np.array(retval).tobytes()).hexdigest()
    def compare(self,funcpy,argspy,funcrust,argsrust):
        try:
            py=getattr(self.vlsvobj_python,funcpy)
            retval_py=py(**argspy)

            rust=getattr(self.vlsvobj_rust,funcrust)
            retval_rust=rust(**argsrust)

        except Exception as e:
            raise e

        if type(retval_py) is dict and type(retval_rust) is dict :
            stack=list(retval_rust.keys())
            if (len(retval_py)!=len(retval_rust)) and len(list(retval_py.keys()))!=0:
                raise SystemError
            for key in retval_py.keys():
                if retval_rust[key] == retval_py[key]:
                    stack.remove(key) #maybe a some ohter way to remove it is faster? 
                else:
                    raise SystemError
            if len(stack)!=0:
                raise SystemError    
            return True
        else:
            raise NotImplementedError 
        

#read ref from file
#(Cellid as variable) reading single cellid ,reading cellid list, (input and output cellid should be same assertion).
#vector (tensor variables from datareduction), read_variable, cellid 0 is nonexistant, error check.
#some datareduction
#read_vdf() (vlsvrs)
# read_variable, compare against fsgird, vg 
# read_velocitycells (vlsvreader)

# read_vdf_spares 
ciTester = Tester("")
for file in files:

    filename=os.path.join(datalocation,file)
    ciTester.changeFile(filename)
    ciTester.load()
    cid=ciTester.vlsvobj_python.get_cellid_with_vdf(np.array([0,0,0]))
    ciTester.compare("read_velocity_cells",{"cellid":cid,"pop":"proton"},"read_vdf_sparse",{"cid":cid,"pop":"proton"})
    quit()

