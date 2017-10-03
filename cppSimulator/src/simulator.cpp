#include "simulator.h"

static int
cppSimulator_init(cppSimulatorObject* self, PyObject *args, PyObject *kwds) {
    return 0;
}

static PyObject*
cppSimulator_get_time(cppSimulatorObject* self) {
    return PyLong_FromLong(0L);
}

static PyMethodDef cppSimulator_methods[] = {
    { "get_time", (PyCFunction)cppSimulator_get_time, METH_NOARGS,
    "get time of simulator"
    },
    { NULL }  /* Sentinel */
};

static PyTypeObject cppSimulatorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cppSimulator",             /* tp_name */
    sizeof(cppSimulatorObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
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
    0,                         /* tp_new */
};

static PyModuleDef cppSimulatorModule = {
    PyModuleDef_HEAD_INIT,
    "cppSimulator module",
    "cppSimulator module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC
PyInit_cppSimulator(void)
{
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
