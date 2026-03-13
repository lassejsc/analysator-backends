/*
 gcc -L. -Wl,-rpath=. -lvlsvrs -ldl -lpython3.10 test.cpp
*/

#include <cassert>
#include <cstddef>
#include <memory>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "stdlib.h"
#include "vlsvrs.h"
#include <Python.h>
#include <cstdint>
#include <iostream>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <set>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>
using namespace std;

// C's % operator is "remainder" operator not modulus like Python's % (not sure
// if there is implementation of this in stnadard library, but its not big so
// here it is)
constexpr int mod(int a, int b) noexcept { return ((a % b) + b) % b; }
constexpr int CHILDS = 8;
constexpr int floordiv(int a, int b) noexcept {
  int q = a / b;
  int r = a % b;
  if ((r != 0) && ((r > 0) != (b > 0))) {
    q--;
  }
  return q;
}

// Convert unordered_map into a Python dictionary
static PyObject *convertToDict(const unordered_map<int, uint64_t> &map) {
  PyObject *dict = PyDict_New();
  for (const auto &it : map) { // structured binding?
    PyObject *key = PyLong_FromLongLong(it.first);
    PyObject *val = PyLong_FromLongLong(it.second);
    PyDict_SetItem(dict, key, val); // error code handling?
    Py_DECREF(key);
    Py_DECREF(val);
  }
  return dict;
}

// could be changed to span?
static void children(int cid, int level, const vector<int64_t> &cid_offsets,
                     const vector<int64_t> &xcells,
                     const vector<int64_t> &ycells,
                     const vector<int64_t> &zcells, vector<int64_t> &out,
                     const vector<vector<int32_t>> &delta) {
  long cellid = cid - 1 - cid_offsets[level];
  vector<int32_t> cellind(3, -1); // could be reused
  cellind[0] = mod(cellid, (xcells[level])) * 2;
  cellind[1] = mod(floordiv(cellid, xcells[level]), (ycells[level])) * 2;
  cellind[2] = floordiv(cellid, xcells[level] * ycells[level]) * 2;
  // children vector always size 8 (hopefully)
  for (size_t i = 0; i < CHILDS; ++i) {
    out[i] =
        cid_offsets[level + 1] + (cellind[0] + delta[i][0]) +
        xcells[level + 1] * (cellind[1] + delta[i][1]) +
        (cellind[2] + delta[i][2]) * xcells[level + 1] * ycells[level + 1] + 1;
  }

  // return out;
}

// Convert python dictionary to unordered_map
static int convertToUnordMap(PyObject *dict,
                             unordered_map<int, uint64_t> &map) {
  PyObject *key, *value;

  Py_ssize_t pos = 0;

  while (PyDict_Next(dict, &pos, &key, &value)) {

    uint64_t val = PyLong_AsUnsignedLongLong(value);
    long keyval = PyLong_AsLong(key);
    map[keyval] = val;
    if (PyErr_Occurred()) {
      return 1;
    }
  }
  return 0;
}

static PyObject *pyBuildDescriptor(PyObject *self, PyObject *args) {
  double xmin, xmax, ymin, ymax, zmin, zmax;
  double dx, dy, dz, xcells, ycells, zcells;
  char *fname;
  // O!|OO (1 required arg (PythonObject) with 2 optional (not sure why we need
  // ! on the first one))
  //  more args O!|O|i for integer
  if (!PyArg_ParseTuple(args, "s", &fname)) {
    return NULL;
  }
  xmin = read_scalar_parameter(fname, "xmin");
  xmax = read_scalar_parameter(fname, "xmax");
  ymin = read_scalar_parameter(fname, "ymin");
  ymax = read_scalar_parameter(fname, "ymax");
  zmin = read_scalar_parameter(fname, "zmin");
  zmax = read_scalar_parameter(fname, "zmax");
  xcells = read_scalar_parameter(fname, "xcells_ini");
  ycells = read_scalar_parameter(fname, "ycells_ini");
  zcells = read_scalar_parameter(fname, "zcells_ini");

  dx = (xmax - xmin) / xcells;
  dy = (ymax - ymin) / ycells;
  dz = (zmax - zmin) / zcells;
  unordered_map<size_t, size_t> fileindex_for_cellid;
  VLSVRS_GenericGrid cellids = read_var_raw(fname, "CellID");
  cout << *(size_t *)cellids.data << " and " << cellids.datasize << endl;
  for (size_t i = 0; i < cellids.nx; ++i) {
    size_t cellid = *(size_t *)(cellids.data + i * cellids.datasize);
    fileindex_for_cellid[cellid] = i;
  }

  cout << cellids.nx << endl;
  cout << fileindex_for_cellid[1] << endl;
  cout << fileindex_for_cellid[2] << endl;
  free(cellids.data);
  return Py_BuildValue("f", dx);
}

template <typename T> struct Npy_Array {
  size_t size;
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  npy_intp *strideptr, *innersizeptr;
  PyArray_Descr **descrGet;
  char **dataptr;
  int typenum = -1;
  Npy_Array(PyArrayObject *arrayin) {
    //
    // Iterator
    typenum = 0;
    iter = NpyIter_New(
        arrayin, NPY_ITER_READONLY | NPY_ITER_REFS_OK | NPY_ITER_EXTERNAL_LOOP,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    iternext = NpyIter_GetIterNext(iter, NULL);
    descrGet = NpyIter_GetDescrArray(iter);
    // the pointer this points to is the actual numpy array and should not be
    // freed here anywhere.
    dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    typenum = (*descrGet)->type_num;
  }
  ~Npy_Array<T>() {
    NpyIter_Deallocate(iter);
    // *dataptr = NULL;
    // strideptr = NULL;
    // innersizeptr = NULL;
    // *descrGet = NULL;
  }
  T *operator[](int i) {

    assert(i < *innersizeptr && i >= 0);
    return (T *)(*dataptr + *strideptr * i);
  }
};
template <typename T> struct Array1 {

  size_t size;
  npy_intp stride;
  T *dataptr;
  Array1(T *dataptr, size_t size, size_t stride)
      : dataptr(dataptr), size(size), stride(stride) {}

  T operator[](int i) const {
    assert((i >= 0 && i < (int)size));
    // cout << "in array1 " << (dataptr + stride / sizeof(T)) << " " << stride
    //      << " " << dataptr << " sizeof " << sizeof(T) << endl;
    return *(T *)(dataptr + i * stride / sizeof(T));
  }
  T &operator[](int i) {
    assert((i >= 0 && i < (int)size));
    // cout << "in array1 " << (dataptr + stride / sizeof(T)) << " " << stride
    //      << " " << dataptr << " sizeof " << sizeof(T) << endl;
    return *(T *)(dataptr + i * stride / sizeof(T));
  }
};
template <typename T> T dotProduct(const Array1<T> &a1, const Array1<T> &a2) {
  // Check that both are same size
  T ret = 0;
  assert(a1.size == a2.size);
  for (size_t i = 0; i < a1.size; ++i) {
    assert(i < a1.size);
    T *val1 = (a1.dataptr + a1.stride / sizeof(T) * i);
    T *val2 = (a2.dataptr + a2.stride / sizeof(T) * i);
    ret = ret + (*val1) * (*val2);
  }
  return ret;
};
template <typename T> struct Array2 : Npy_Array<T> {
  size_t size;
  npy_intp dim0;
  npy_intp dim1;
  // T *arrptr;
  size_t strideouter; // note currenty if array changes this may cause issues if
  // value wqs cop8edi
  Array2(PyArrayObject *arrin) : Npy_Array<T>(arrin) {
    size = PyArray_SIZE(arrin);
    dim0 = PyArray_DIM(arrin, 0);
    dim1 = PyArray_DIM(arrin, 1);
    assert(PyArray_NDIM(arrin) == 2);

    strideouter = PyArray_STRIDE(arrin, 0);
  }
  vector<T> getCopy(int i) const {
    assert(i >= 0 && i < dim0);
    cout << "vec func" << endl;
    vector<T> ret;
    do {
      char *data = *this->dataptr;
      npy_intp stride = *this->strideptr;
      npy_intp count = dim1;
      while (count--) {
        ret.push_back(*(int *)data);
        data += stride;
      }
    } while (this->iternext(this->iter));
    return ret;
  }
  Array1<T> operator[](int i) {
    assert(i >= 0 && i < dim0);
    // return (T *)(*dataptr + *strideouter * i);
    // cout << " INSIDE ARR2D " << (T *)(*this->dataptr + *this->strideptr)
    //      << " " << (T *)(*this->strideptr) << " " << (T *)(*this->dataptr)
    //      << " end" << endl;
    //      << *this->strideptr << endl;
    return Array1<T>((T *)(*this->dataptr + i * strideouter), dim1,
                     *this->strideptr);
  }
};
// Make new struct for 2D array, make it check that shape is currect
// we may need pointer to the memory addrs of axis then the stride in there
// which is given by PyArray_STRIDE(array,0)
// Then dotproduct with these (ptr,stride) info for two arrays ()

// We need build_cell_neighborhood, get_cell_corner_vertices,
// build_dual_from_vertices

static PyObject *pyHandleCell(PyObject *self, PyObject *args) {
  PyObject *fileindex_for_cellid;
  PyArrayObject *array;
  // O!|OO (1 required arg (PythonObject) with 2 optional (not sure why we need
  // ! on the first one))
  //  more args O!|O|i for integer
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
    return NULL;
  }
  cout << PyArray_SIZE(array) << endl;
  cout << PyArray_STRIDE(array, 0) << endl;
  cout << *PyArray_SHAPE(array) << endl;
  Array2<int> arra(array);
  cout << "------------------" << endl;
  cout << arra[0][0] << endl;
  cout << arra[0][1] << endl;
  cout << arra[0][2] << endl;
  cout << arra[0][3] << endl;
  cout << arra[0][4] << endl;

  cout << "------------------" << endl;
  cout << arra[1][0] << endl;
  cout << arra[1][1] << endl;
  cout << arra[1][2] << endl;
  cout << arra[1][3] << endl;
  cout << arra[1][4] << endl;
  cout << dotProduct(arra[0], arra[0]) << endl;
  set<uint32_t> cells_todo_set;
  // we would need f.build_cellneighborhood function to make it into C, wihch
  // means we woul dneed to make the whole vlsvreader(?) lol
  uint32_t cid0 = 303563682;
  cells_todo_set.insert(cid0);

  cout << "------------------" << endl;
  vector<int> t = arra.getCopy(0);
  for (auto a : t) {
    cout << a << endl;
  }
  // vector<char*> read_Array;
  // do {
  //   /* Get the inner loop data/stride/count values */
  //   char *data = *dataptr;
  //
  //
  //   npy_intp stride = *strideptr;
  //   npy_intp count = *innersizeptr;
  //   // assert(strideptr[1] == sizeof(double)); // so data++ would work
  //   /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
  //
  //   while (count--) {
  //     // read_Array.push_back(data);
  //     data += stride;
  //   }
  //   cout << "uhhhh" << endl;
  //   /* Increment the iterator to the next inner loop */
  // } while (iternext(iter));
  // NpyIter_Deallocate(iter);
  printf("test\n");
  // main loop proto
  // uint_32 maxiters=2000000;
  // uint_32 icount=0;
  // while((icount < maxiters) && (cells_todo.size()>0)){
  // Do something
  //    icount++;
  //    if (icount==maxiters){
  //            err << "max iters"<<endl;
  //    }
  //
  //  }
  // npy_intp outshape;
  // NpyIter_GetShape(iter, &outshape);
  // cout << outshape << endl;
  // cout << NpyIter_GetIterSize(iter) << endl;
  // cout << (*descrGet)->type_num << endl;
  // Descriptor for the datatype
  // PyArray_Descr *reqDescr = PyArray_DescrFromType(NPY_DOUBLE);
  // PyArrayObject *arr = (PyArrayObject *)PyArray_FromArray(
  //    array, reqDescr,
  //    NPY_ARRAY_CARRAY); // Not sure exactly what happens but I believe this
  // return the pointer to the original array as using
  // dataPtr below to modify a value WILL modify original
  // array in python
  // double *dataPtr =
  //     static_cast<double *>(PyArray_DATA(arr)); // to pointer or not to?
  // NpyIter_Deallocate(iter);
  // for (size_t i = 0; i < 10; ++i) {
  //   cout << dataPtr[i] << endl;
  // }

  // for (auto *dat : dataPtr) {
  //   cout << dat << endl;
  // }

  // if (convertToUnordMap(fileindex_for_cellid, fileindex_for_cellid_map) != 0)
  // {
  //   cout << "error" << endl;
  // }
  // PyObject *outdict = convertToDict(idxToFileIndex);
  // return Py_BuildValue("sO", descr.str().data(), outdict);
  return Py_BuildValue("i", 0);
}

static PyMethodDef cpphelpers_methods[] = {
    {"buildDescriptor", pyBuildDescriptor, METH_VARARGS, "Build descriptor"},
    {"test", pyHandleCell, METH_VARARGS, "test"},
    {NULL} /* Sentinel */
};
static struct PyModuleDef cpphelpers = {
    .m_methods = cpphelpers_methods,
};
PyMODINIT_FUNC PyInit_cpphelpers(void)
// create the module
{
  import_array();
  return PyModuleDef_Init(&cpphelpers);
}
