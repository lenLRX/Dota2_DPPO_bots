#include "simulator.h"
#include "simulatorImp.h"
#include "log.h"

#define __ENGINE_VERSION__ "0.0.6"

static void
cppSimulator_dealloc(cppSimulatorObject* self)
{
    //printf("going to delete pImp\n");
    delete self->pImp;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
cppSimulator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    cppSimulatorObject *self;
    self = (cppSimulatorObject *)type->tp_alloc(type, 0);
    //Py_XDECREF(args);
    //Py_XDECREF(kwds);
    return (PyObject *)self;
}

static int
cppSimulator_init(cppSimulatorObject* self, PyObject *args, PyObject *kwds) {
    //printf("initing simulator\n");
    PyObject* obj_canvas = NULL;
    if (!PyArg_ParseTuple(args, "|O",&obj_canvas)) {
        return -1;
    }
    if (obj_canvas == Py_None) {
        obj_canvas = NULL;
        Py_DECREF(Py_None);
    }
    self->pImp = new cppSimulatorImp(self, obj_canvas);
    //Py_XDECREF(args);
    //Py_XDECREF(kwds);
    return 0;
}

static PyObject*
cppSimulator_get_time(cppSimulatorObject* self) {
    cppSimulatorImp* pImp = self->pImp;
    return PyFloat_FromDouble(pImp->get_time());
}

static PyObject*
cppSimulator_loop(cppSimulatorObject* self) {
    self->pImp->loop();
//https://stackoverflow.com/questions/15287590/why-should-py-increfpy-none-be-required-before-returning-py-none-in-c
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
cppSimulator_set_order(cppSimulatorObject* self, PyObject *args, PyObject *kwds) {
    self->pImp->set_order(args, kwds);
    //Py_XDECREF(args);
    //Py_XDECREF(kwds);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
cppSimulator_get_state_tup(cppSimulatorObject* self, PyObject *args, PyObject *kwds) {
    char* side = NULL;
    int idx = 0;
    if (!PyArg_ParseTuple(args, "si", &side, &idx)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    //Py_XDECREF(args);
    //Py_XDECREF(kwds);
    return self->pImp->get_state_tup(side, idx);
}

static PyObject*
cppSimulator_predefined_step(cppSimulatorObject* self, PyObject *args, PyObject *kwds) {
    char* side = NULL;
    int idx = 0;
    if (!PyArg_ParseTuple(args, "si", &side, &idx)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    //Py_XDECREF(args);
    //Py_XDECREF(kwds);
    return self->pImp->predefined_step(side, idx);
}

static PyObject*
cppSimulator_get_version(cppSimulatorObject* self) {
    return Py_BuildValue("s", __ENGINE_VERSION__);
}

static PyMethodDef cppSimulator_methods[] = {
    { "get_time", (PyCFunction)cppSimulator_get_time, METH_NOARGS,
    "get time of simulator"
    },
    { "loop", (PyCFunction)cppSimulator_loop, METH_NOARGS,
    "mainloop of simulator"
    },
    { "get_state_tup", (PyCFunction)cppSimulator_get_state_tup, METH_VARARGS | METH_KEYWORDS,
    "get (state, reward, done) by side and idx"
    },
    { "predefined_step", (PyCFunction)cppSimulator_predefined_step, METH_VARARGS | METH_KEYWORDS,
    "get predefined_step by side and idx"
    }
    ,
    { "set_order", (PyCFunction)cppSimulator_set_order, METH_VARARGS | METH_KEYWORDS,
    "set move order side,idx,x,y"
    },
    { "get_version", (PyCFunction)cppSimulator_get_version, METH_NOARGS,
    "get version of simulator"
    },
    { NULL }  /* Sentinel */
};

static PyTypeObject cppSimulatorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cppSimulator.cppSimulator",             /* tp_name */
    sizeof(cppSimulatorObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)cppSimulator_dealloc,/* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,       /* tp_flags */
    "cppSimulatorObject",      /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    cppSimulator_methods,      /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)cppSimulator_init,      /* tp_init */
    0,                         /* tp_alloc */
    cppSimulator_new,          /* tp_new */
};

static PyModuleDef cppSimulatorModule = {
    PyModuleDef_HEAD_INIT,
    "cppSimulator",
    "cppSimulator module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC
PyInit_cppSimulator(void)
{
    Logger::getInstance().redirectStream("log/cpplog.txt");
    PyObject* m;

    cppSimulatorType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&cppSimulatorType) < 0)
        return NULL;

    m = PyModule_Create(&cppSimulatorModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&cppSimulatorType);
    PyModule_AddObject(m, "cppSimulator", (PyObject *)&cppSimulatorType);
    return m;
}
