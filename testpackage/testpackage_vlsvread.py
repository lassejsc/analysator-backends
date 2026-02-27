import analysator as pt
import numpy as np
import vlsvrs
import os
import hashlib
import pickle

datalocation = "/turso/group/spacephysics/analysator/CI/analysator-test-data/vlasiator/"
files=["3D/FID/bulk1/bulk1.0000995.vlsv",
       "3D/FHA/bulk1/bulk1.0000990.vlsv",
       "2D/BCQ/bulk/bulk.002002.vlsv",
       "2D/ABC/bulk.0001003.vlsv"
]
# "2D/BCQ/bulk/bulk.0002002.vlsv",
# filename='/home/siclasse/Downloads/bulk_hermite_compressed.0000001.vlsv'
# filename='/home/siclasse/bulk.0000110.vlsv'
class Tester:
    def __init__(self,filename=None):
        self.filename=filename
        self.vlsvobj=None
        self.hashes_dict_rust={}
        self.hashes_dict_python={}
    def changeFile(self,filename):
        self.filename=filename
    def loadPickle(self,file):
        self.pickled = pickle.load(file)
    def dumpPickle(self,file):
        pickle.dump(self.hashes_dict,file)

    def load(self,obj=None):
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
    def hash(self,func,args,op=None,opargs=None,both=False):
        
        def update(vlsvobj,op,opargs,args):
            try:
                t=getattr(vlsvobj,func)
                if type(args) is dict:
                    retval=t(**args)
                elif type(args) is list:
                    retval=t(*args)
                if op and opargs:
                    if type(op) is not list:
                       op=[op]
                       opargs=[opargs]
                    if type(op) is list:
                        for i,f in enumerate(op):
                            try:
                                fun=getattr(retval,f)
                            except AttributeError:
                                try: 
                                    if '.' in f:
                                        funcl=f.split('.')
                                        if type(funcl[0]) is str:
                                            import importlib
                                            funcl[0]=importlib.import_module(funcl[0])
                                        fun=getattr(funcl[0],funcl[1])
                                    else:
                                        fun=f
                                    opargs[i]=[retval]
                                except AttributeError as e:
                                    raise AttributeError(f"Did not find func {func} to operate with: {e}")
                            retval=fun(*opargs[i])
            except Exception as e:
                raise e
            if func not in self.hashes_dict_rust.keys():
                self.hashes_dict_rust[func]={}
            elif func not in self.hashes_dict_python.keys():
                self.hashes_dict_python[func]={}
            try:
                self.hashes_dict_rust[func][vlsvobj.get_filename()]=hashlib.sha256(np.array(retval).tobytes()).hexdigest()
            except AttributeError:
                self.hashes_dict_python[func][vlsvobj.file_name]=hashlib.sha256(np.array(retval).tobytes()).hexdigest()

        if not both:
            update(self.vlsvobj,op,opargs,args)
        else:
            update(self.vlsvobj_rust,op,opargs,args)
            update(self.vlsvobj_python,op,opargs,args)

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
                raise SystemError("one or both of the dictionaries returned by the readers are empty")
            for key in retval_py.keys():
                if retval_rust[key] == retval_py[key]:
                    stack.remove(key) #maybe a some ohter way to remove it is faster? 
                else:
                    raise SystemError("returned dictionary values between vlsvreader and vlsvrs do not match")
            if len(stack)!=0:
                raise KeyError("returned dictionry from vlsvrs contains keys not present in dictonary returned by python.") 
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
# 
ciTester = Tester()
for file in files:

    #Load data 
    filename=os.path.join(datalocation,file)

    #filename="/home/siclasse/bulk.0000110.vlsv"
    #filename="/home/siclasse/Downloads/bulk_hermite_compressed.0000001.vlsv"
    ciTester.changeFile(filename)
    ciTester.load()

    #Test compare
    cid=ciTester.vlsvobj_python.get_cellid_with_vdf(np.array([0,0,0]))
    ciTester.compare("read_velocity_cells",{"cellid":cid,"pop":"proton"},"read_vdf_sparse",{"cid":cid,"pop":"proton"})
    
    
    #Make hash python
    ciTester.setHashTarget("rust")
    ciTester.hash("read_variable",{"variable":"CellID","op":0},op=["reshape","astype","numpy.sort"],opargs=[[tuple([-1])],[int],[]])
    ciTester.setHashTarget("python")
    ciTester.hash("read_variable",["CellID"])

print(ciTester.hashes_dict_python)
print(ciTester.hashes_dict_rust)


