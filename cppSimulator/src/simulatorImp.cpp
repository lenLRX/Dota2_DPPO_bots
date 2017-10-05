#include "simulatorImp.h"
#include "Config.h"
#include "Event.h"

cppSimulatorImp::cppSimulatorImp(cppSimulatorObject* obj, PyObject* canvas)
    :self(obj), tick_time(0.0), tick_per_second(Config::tick_per_second), 
    delta_tick(1.0 / Config::tick_per_second), canvas(canvas)
{
    Py_INCREF(self);
    Py_XINCREF(canvas);
    EventFactory::CreateSpawnEvnt(this);
}

cppSimulatorImp::~cppSimulatorImp()
{
    for (auto p : allSprites) {
        delete p;
    }
    Py_DECREF(self);
    Py_XDECREF(canvas);
}

double cppSimulatorImp::get_time()
{
    return tick_time;
}

void cppSimulatorImp::loop()
{
    tick_tick();
    while (!queue.empty() &&
        queue.top().get_time() < tick_time)
    {
        auto event = queue.top();
        queue.pop();
        event.activate();
    }

    for (Sprite* s : Sprites) {
        s->step();
    }
}
