#include <Python.h>

typedef struct {
    PyObject_HEAD
} cppSimulatorObject;

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
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "cppSimulatorObject",           /* tp_doc */
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