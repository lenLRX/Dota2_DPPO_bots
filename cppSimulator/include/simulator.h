#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <Python.h>

class cppSimulatorImp;

typedef struct {
    PyObject_HEAD
    cppSimulatorImp* pImp;
} cppSimulatorObject;

#endif//__SIMULATOR_H__